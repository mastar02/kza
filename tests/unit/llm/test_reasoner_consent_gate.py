"""OQ-8: HTTP reasoners stop egress, retries, streams and late replies on withdrawal."""
import inspect
import json
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from src.core.egress import CloudConsentGate, ConsentWithdrawalSource, EgressDenied, EgressPolicy
from src.llm import reasoner as reasoner_module
from src.llm.error_classifier import classify_error
from src.llm.http_generation import HTTPGenerationClient
from src.llm.reasoner import FastRouter, HttpReasoner
from src.llm.types import ErrorKind
from tests.mocks.llm_http import SSEBody, event
from tests.unit.llm.test_http_async_stream import setup

CLOUD = "https://gateway.test/v1"
LOCAL = "http://127.0.0.1:8101/v1"


def withdraw(gate):
    gate.withdraw(ConsentWithdrawalSource.ACTOR)


@pytest.fixture
def transport(monkeypatch):
    """Real httpx.Client + real OpenAI SDK over an in-memory transport."""
    state = SimpleNamespace(requests=[], on_post=None)

    def handle(request):
        state.requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "synthetic-model", "object": "model", "created": 0, "owned_by": "test"}]})
        if state.on_post is not None:
            return state.on_post(request)
        return httpx.Response(200, json={"id": "x", "object": "text_completion", "created": 0,
            "model": "synthetic-model", "choices": [{"index": 0, "text": "hola", "finish_reason": "stop",
                                                     "logprobs": None}]})

    actual = httpx.Client

    class Client(actual):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=httpx.MockTransport(handle), **kwargs)

    class HttpxProxy:
        def __getattr__(self, name):
            return Client if name == "Client" else getattr(httpx, name)

    monkeypatch.setattr(reasoner_module, "httpx", HttpxProxy())
    monkeypatch.setattr(reasoner_module, "_resolve_api_key", lambda *args, **kwargs: "synthetic-key")
    return state


# ----------------------------------------------------------------------
# M1: no SDK retry after withdrawal; per-request hook on every client.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("factory", ["http", "fast"])
@pytest.mark.parametrize("destination", ["cloud", "local"])
def test_openai_max_retries_zero_only_for_cloud(monkeypatch, factory, destination):
    seen = []

    class SDK:
        def __init__(self, **kwargs):
            seen.append(kwargs)
            self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="m")]))

        def close(self):
            kwargs = seen[-1]
            kwargs["http_client"].close()
    monkeypatch.setattr(openai, "OpenAI", SDK)
    monkeypatch.setattr(reasoner_module, "_resolve_api_key", lambda *args, **kwargs: "synthetic-key")
    url = CLOUD if destination == "cloud" else LOCAL
    if factory == "http":
        client = HttpReasoner(base_url=url, model="m", destination=destination, cloud_consent=True,
                              consent_gate=CloudConsentGate())
    else:
        client = FastRouter(base_url=url, model="m", destination=destination, cloud_consent=True,
                            consent_gate=CloudConsentGate())
    client.load()
    try:
        if destination == "cloud":
            assert seen[-1]["max_retries"] == 0
        else:
            assert "max_retries" not in seen[-1]
        hooks = seen[-1]["http_client"].event_hooks["request"]
        assert hooks and not any(inspect.iscoroutinefunction(hook) for hook in hooks)
    finally:
        client.close() if factory == "http" else seen[-1]["http_client"].close()


@pytest.mark.parametrize("factory", ["http", "fast"])
def test_sync_hook_allows_then_denies(transport, factory):
    gate = CloudConsentGate()
    if factory == "http":
        client = HttpReasoner(base_url=CLOUD, model="synthetic-model", cloud_consent=True, consent_gate=gate)
    else:
        client = FastRouter(base_url=CLOUD, model="synthetic-model", cloud_consent=True, consent_gate=gate)
    try:
        client.load()  # Allowed: the hook passes the open request through.
        assert [r.url.path for r in transport.requests] == ["/v1/models"]
        withdraw(gate)
        with pytest.raises(openai.APIConnectionError) as caught:
            client._client.models.list()  # Bypasses every explicit check: only the hook remains.
        assert classify_error(caught.value) is ErrorKind.CONSENT_WITHDRAWN
        assert len(transport.requests) == 1
    finally:
        if factory == "http":
            client.close()
        else:
            client._client.close()


def test_failed_first_attempt_is_never_retried_after_withdrawal(transport):
    gate = CloudConsentGate()
    client = HttpReasoner(base_url=CLOUD, model="synthetic-model", cloud_consent=True, consent_gate=gate)

    def fail_and_withdraw(request):
        withdraw(gate)
        return httpx.Response(503, json={"error": {"message": "synthetic overload"}})
    transport.on_post = fail_and_withdraw
    try:
        client.load()
        with pytest.raises(Exception):
            client.generate("consulta")
        assert [r.method for r in transport.requests] == ["GET", "POST"]
    finally:
        client.close()


async def test_async_hook_is_a_coroutine_and_allows_open_requests():
    gate = CloudConsentGate()
    policy = EgressPolicy.for_http(CLOUD, "cloud", True, consent_gate=gate)
    requests = []

    async def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": "m"}]})
    raw = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    client = HTTPGenerationClient(CLOUD, api_key="k", http_client=raw, request_guard=policy.require_allowed)
    try:
        hooks = raw.event_hooks["request"]
        assert hooks and all(inspect.iscoroutinefunction(hook) for hook in hooks)
        client.prepare("chat")
        assert await client.models() == ["m"]
        withdraw(gate)
        client.prepare("chat")
        with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
            await client.models()
        assert len(requests) == 1
    finally:
        await client.close()


async def test_async_hook_keeps_existing_hooks_on_an_injected_client():
    seen = []

    async def existing(request):
        seen.append("existing")
    raw = httpx.AsyncClient(event_hooks={"request": [existing]},
                            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []})))
    client = HTTPGenerationClient(CLOUD, api_key="k", http_client=raw, request_guard=lambda: seen.append("guard"))
    try:
        client.prepare("chat")
        await client.models()
        assert seen == ["existing", "guard"]
    finally:
        await client.close()


async def test_internal_async_client_gets_the_coroutine_hook():
    client = HTTPGenerationClient(CLOUD, api_key="k", request_guard=lambda: None)
    try:
        hooks = client._client.event_hooks["request"]
        assert len(hooks) == 1 and inspect.iscoroutinefunction(hooks[0])
    finally:
        await client.close()


def test_reasoner_passes_its_policy_guard_to_the_async_client(monkeypatch):
    seen = []

    class Capture:
        def __init__(self, *args, **kwargs):
            seen.append(kwargs)
    monkeypatch.setattr(reasoner_module, "HTTPGenerationClient", Capture)
    monkeypatch.setattr(reasoner_module, "_resolve_api_key", lambda *args, **kwargs: "k")
    reasoner = HttpReasoner(base_url=CLOUD, cloud_consent=True, consent_gate=CloudConsentGate())
    reasoner._new_async_client(CLOUD)
    assert seen[0]["request_guard"] == reasoner.policy.require_allowed


# ----------------------------------------------------------------------
# m5 + F4: streams cut before the next yield; late replies discarded.
# ----------------------------------------------------------------------

class _WithdrawingBody(SSEBody):
    """Withdraw just as the transport delivers the given part index."""

    def __init__(self, parts, gate, at):
        super().__init__(parts)
        self.gate, self.at = gate, at

    async def __aiter__(self):
        for index, part in enumerate(self.parts):
            if index == self.at:
                withdraw(self.gate)
            yield part


LONG = "Una respuesta suficientemente larga para pasar el holdback. "


async def test_astream_never_yields_after_withdrawal(monkeypatch):
    gate = CloudConsentGate()
    reasoner, endpoint = setup(monkeypatch, consent_gate=gate)
    endpoint.body = _WithdrawingBody([event(LONG), event(LONG), event(finish="stop")], gate, 0)
    chunks = []
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
        async with aclosing(reasoner.astream("consulta")) as stream:
            async for chunk in stream:
                chunks.append(chunk)
    assert chunks == []
    assert endpoint.body.closed and endpoint.client.is_closed()


async def test_astream_withdrawal_between_chunks_stops_the_stream(monkeypatch):
    gate = CloudConsentGate()
    reasoner, endpoint = setup(monkeypatch, consent_gate=gate)
    endpoint.body = SSEBody([event(LONG), event(LONG), event(finish="stop")])
    chunks = []
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
        async with aclosing(reasoner.astream("consulta")) as stream:
            async for chunk in stream:
                chunks.append(chunk)
                withdraw(gate)
    assert len(chunks) == 1 and endpoint.body.closed and endpoint.client.is_closed()


def _stream_client(pieces, gate, at):
    chunks = []
    for piece in pieces:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = piece
        chunks.append(chunk)

    def iterate():
        for index, chunk in enumerate(chunks):
            if index == at:
                withdraw(gate)
            yield chunk
    client = MagicMock()
    client.chat.completions.create.return_value = iterate()
    return client


def test_generate_stream_final_holdback_not_flushed_after_withdrawal():
    gate = CloudConsentGate()
    reasoner = HttpReasoner(base_url=CLOUD, api_style="chat", cloud_consent=True, consent_gate=gate)
    reasoner._client = _stream_client(["ho", "la"], gate, 1)  # Shorter than the holdback: only the final flush.
    reasoner._resolved_model = "m"
    out = []
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
        for record in reasoner.generate_stream("hi"):
            out.append(record)
    assert out == []


def test_generate_stream_per_chunk_yield_is_checked():
    gate = CloudConsentGate()
    reasoner = HttpReasoner(base_url=CLOUD, api_style="chat", cloud_consent=True, consent_gate=gate)
    reasoner._client = _stream_client([LONG, LONG], gate, 0)
    reasoner._resolved_model = "m"
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
        assert list(reasoner.generate_stream("hi")) == []


def test_late_sync_reply_is_discarded():
    gate = CloudConsentGate()
    reasoner = HttpReasoner(base_url=CLOUD, api_style="chat", cloud_consent=True, consent_gate=gate)
    client = MagicMock()

    def create(**kwargs):
        withdraw(gate)
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "respuesta tardía"
        return response
    client.chat.completions.create.side_effect = create
    reasoner._client, reasoner._resolved_model = client, "m"
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
        reasoner.generate("hi")


async def test_late_async_nonstream_reply_is_discarded(monkeypatch):
    gate = CloudConsentGate()
    reasoner, endpoint = setup(monkeypatch, consent_gate=gate)
    handle = endpoint.handle

    async def late(request):
        if request.method == "POST":
            withdraw(gate)
            endpoint.requests.append(request)
            return httpx.Response(200, json={"choices": [{"index": 0, "finish_reason": "stop",
                                                          "message": {"content": "respuesta tardía"}}]})
        return await handle(request)
    monkeypatch.setattr(endpoint, "handle", late)
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
        await reasoner.complete("consulta")
    assert json.loads(endpoint.requests[-1].content)["stream"] is False


def test_fast_router_late_reply_is_discarded():
    gate = CloudConsentGate()
    router = FastRouter(base_url=CLOUD, cloud_consent=True, consent_gate=gate)
    client = MagicMock()

    def create(**kwargs):
        withdraw(gate)
        return SimpleNamespace(choices=[SimpleNamespace(text="tarde")], usage=None)
    client.completions.create.side_effect = create
    router._client = client
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
        router._complete_native("hi", 8, 0.1, None)
    router2 = FastRouter(base_url=CLOUD, cloud_consent=True, consent_gate=CloudConsentGate())
    client2 = MagicMock()

    def create2(**kwargs):
        withdraw(router2.policy.consent_gate)
        return SimpleNamespace(choices=[SimpleNamespace(text="tarde")])
    client2.completions.create.side_effect = create2
    router2._client = client2
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
        router2.generate(["hi"])


def test_local_router_ignores_a_withdrawn_gate():
    gate = CloudConsentGate()
    withdraw(gate)
    router = FastRouter(base_url=LOCAL, destination="local", consent_gate=gate)
    client = MagicMock()
    client.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(text="ok")], usage=None)
    router._client = client
    assert router._complete_native("hi", 8, 0.1, None) == "ok"


async def test_async_hook_is_installed_once_per_guard():
    policy = EgressPolicy.for_http(CLOUD, "cloud", True, consent_gate=CloudConsentGate())
    raw = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []})))
    first = HTTPGenerationClient(CLOUD, api_key="k", http_client=raw, request_guard=policy.require_allowed)
    HTTPGenerationClient(CLOUD, api_key="k", http_client=raw, request_guard=policy.require_allowed)
    try:
        assert len(raw.event_hooks["request"]) == 1
    finally:
        await first.close()


async def test_late_reply_after_idle_timeout_stream_join_is_discarded(monkeypatch):
    """F4: the idle-timeout SSE join is re-checked after the stream ends without a final yield."""
    gate = CloudConsentGate()
    reasoner, endpoint = setup(monkeypatch, consent_gate=gate, idle_timeout_s=5)
    endpoint.body = _WithdrawingBody([event("<think>solo razonamiento</think>"), event(finish="stop")], gate, 1)
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
        await reasoner.complete("consulta")
