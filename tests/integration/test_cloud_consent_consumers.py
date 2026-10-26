"""Denied consent must stop real inference consumers before their transport."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import sys

import httpx
import numpy as np
import pytest

from src.ambient.distiller import make_local_chat_fn
from src.code_index.cards import CardGenerator
from src.core.egress import EgressDenied, EgressPolicy
from src.llm.cloud_consent import resolve_compaction_endpoint, resolve_reasoner_gate
from src.llm.hermes_reasoner import HermesCliReasoner
from src.llm.reasoner import FastRouter, HttpReasoner
from src.llm.router_factory import build_llm_router
from src.providers.remote import (
    RemoteLLMProvider, RemoteProviderConfig, RemoteRouterProvider,
    RemoteSTTProvider, RemoteTTSProvider,
)
from tests.mocks.llm_http import Endpoint, SSEBody


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["http://127.0.0.1:8200/v1", "http://localhost:8200/v1", "http://[::1]:8200/v1"])
async def test_http_reasoner_denial_covers_every_api_and_cached_sdk(url):
    reasoner = HttpReasoner(base_url=url, destination="cloud", cloud_consent=False)
    sdk = MagicMock()
    reasoner._client = sdk
    with pytest.raises(EgressDenied):
        reasoner.load()
    with pytest.raises(EgressDenied):
        reasoner.generate("private turn")
    with pytest.raises(EgressDenied):
        list(reasoner.generate_stream("private turn"))
    with pytest.raises(EgressDenied):
        await reasoner.complete("private turn")
    assert sdk.mock_calls == []


@pytest.mark.asyncio
async def test_revoked_consent_is_checked_again_with_a_cached_client():
    reasoner = HttpReasoner(cloud_consent=True)
    sdk = MagicMock()
    reasoner._client = sdk
    reasoner.policy = EgressPolicy("cloud", False, reasoner.base_url)
    with pytest.raises(EgressDenied):
        await reasoner.complete("private turn")
    assert sdk.mock_calls == []


@pytest.mark.asyncio
async def test_fast_router_does_not_infer_locality_from_loopback():
    router = FastRouter(base_url="http://127.0.0.1:8200/v1")
    sdk = MagicMock()
    router._client = sdk
    with pytest.raises(EgressDenied):
        router.generate(["private turn"])
    with pytest.raises(EgressDenied):
        await router.complete("private turn")
    assert sdk.mock_calls == []


@pytest.mark.asyncio
async def test_cli_reasoner_denies_before_subprocess_or_usage_files():
    reasoner = HermesCliReasoner()
    with patch("src.llm.hermes_reasoner.subprocess.run") as status, patch("src.llm.hermes_reasoner.subprocess.Popen") as inference, patch("src.llm.hermes_reasoner.tempfile.mkstemp") as usage:
        with pytest.raises(EgressDenied):
            reasoner.load()
        with pytest.raises(EgressDenied):
            await reasoner.complete("private turn")
        status.assert_not_called()
        inference.assert_not_called()
        usage.assert_not_called()


@pytest.mark.asyncio
async def test_cards_deny_even_with_a_cached_client():
    generator = CardGenerator("http://127.0.0.1:8200/v1", "model")
    sdk = MagicMock()
    generator._client = sdk
    with pytest.raises(EgressDenied):
        await generator.generate("src/example.py", "private source")
    assert sdk.mock_calls == []


@pytest.mark.asyncio
async def test_ambient_distillation_cannot_send_to_an_undeclared_gateway():
    chat = make_local_chat_fn("http://127.0.0.1:8200/v1")
    with patch("aiohttp.ClientSession") as transport:
        with pytest.raises(EgressDenied):
            await chat("private conversation")
        transport.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["stt", "tts", "llm", "router"])
async def test_remote_consumers_default_to_denied_even_over_loopback(kind):
    config = RemoteProviderConfig("http://127.0.0.1:9999")
    providers = {"stt": RemoteSTTProvider, "tts": RemoteTTSProvider, "llm": RemoteLLMProvider, "router": RemoteRouterProvider}
    provider = providers[kind](config)
    if kind == "stt":
        call = provider.transcribe_async(np.zeros(100, dtype=np.float32))
    elif kind == "tts":
        call = provider.synthesize_async("private speech")
    elif kind == "llm":
        call = provider.generate_async("private turn")
    else:
        call = provider.classify_async("private turn", ["command"])
    with patch("aiohttp.ClientSession") as transport:
        with pytest.raises(EgressDenied):
            await call
        transport.assert_not_called()


def test_local_parent_grant_cannot_authorize_a_cloud_compactor_override():
    reasoner = {"http_base_url": "http://127.0.0.1:8101/v1", "destination": "local", "cloud": {"consent": False}, "api_key_env": "CLOUD_KEY"}
    allowed, _ = resolve_reasoner_gate(reasoner, "http", "http://127.0.0.1:8200/v1")
    assert allowed
    endpoint = resolve_compaction_endpoint({"base_url": "http://127.0.0.1:8200/v1", "destination": "cloud"}, reasoner, allowed)
    assert endpoint.base_url == "http://127.0.0.1:8101/v1"
    assert endpoint.destination == "local"
    assert endpoint.api_key_env is None


def test_gateway_fallback_never_inherits_an_unresolved_local_declaration():
    config = {"http_base_url": "${UNSET}", "destination": "local", "cloud": {"consent": True}}
    allowed, url = resolve_reasoner_gate(config, "http", "http://127.0.0.1:8200/v1")
    assert allowed
    assert config["destination"] == "cloud"
    config["cloud"]["consent"] = False
    assert resolve_reasoner_gate(config, "http", url)[0] is False


@pytest.mark.parametrize("client", [HttpReasoner, FastRouter, CardGenerator])
def test_missing_http_url_cannot_select_an_sdk_default_provider(client):
    kwargs = {"base_url": None, "cloud_consent": True}
    if client is CardGenerator:
        kwargs["model"] = "model"
    with pytest.raises(ValueError, match="explicit endpoint"):
        client(**kwargs)


def test_sync_inventory_client_also_denies_before_creating_an_sdk():
    from scripts.sync_ha_to_chroma import VLLMClient

    with patch("openai.OpenAI") as sdk:
        with pytest.raises(EgressDenied):
            VLLMClient("http://127.0.0.1:8200/v1", "model")
        sdk.assert_not_called()


@pytest.mark.asyncio
async def test_inherited_local_compaction_remains_available_without_cloud_consent(monkeypatch):
    endpoint = resolve_compaction_endpoint({}, {"http_base_url": "http://127.0.0.1:8101/v1", "destination": "local", "cloud": {"consent": False}}, True)
    reasoner = HttpReasoner(**{key: getattr(endpoint, key) for key in ("base_url", "destination", "cloud_consent", "api_style")})
    transport = Endpoint(SSEBody([]))
    transport.completion_payload = {"choices": [{"message": {"content": "local answer"}, "finish_reason": "stop"}]}
    monkeypatch.setattr(reasoner, "_new_async_client", transport.build)
    assert await reasoner.complete("private turn") == "local answer"
    assert [request.method for request in transport.requests] == ["GET", "POST"]
    assert str(transport.requests[-1].url) == endpoint.base_url + "/chat/completions"
    assert transport.client.is_closed()


async def test_failover_chain_cannot_override_a_consumer_denial(tmp_path):
    local = FastRouter(destination="local")
    local._client = MagicMock()
    local._client.completions.create.side_effect = TimeoutError("Local timeout")
    cloud = HttpReasoner(base_url="http://127.0.0.1:8200/v1")
    cloud._client = MagicMock()
    router = build_llm_router({
        "endpoints": [
            {"id": "local", "kind": "fast_router", "priority": 1},
            {"id": "cloud", "kind": "http_reasoner", "priority": 2},
        ],
        "cooldowns": {"persist_path": str(tmp_path / "cooldowns.json")},
    }, {"local": local, "cloud": cloud})
    with pytest.raises(EgressDenied):
        await router.complete("private turn")
    local._client.completions.create.assert_called_once()
    assert cloud._client.mock_calls == []


@pytest.mark.parametrize("base_url,destination", [
    ("http://127.0.0.1:8200/v1", "cloud"),
    ("${MISSING_GATEWAY}", "local"),
])
async def test_code_index_factory_preserves_denial_after_endpoint_resolution(tmp_path, monkeypatch, base_url, destination):
    from src.code_index.service import build_indexer

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=MagicMock()))
    monkeypatch.setattr("src.code_index.service.build_chroma_client", MagicMock())
    indexer = build_indexer({
        "cards": {"base_url": base_url, "destination": destination, "model": "model"},
        "manifest_path": str(tmp_path / "manifest.json"),
    }, tmp_path, cloud_consent=False)
    with patch("openai.AsyncOpenAI") as sdk:
        with pytest.raises(EgressDenied):
            await indexer.card_generator.generate("src/example.py", "private source")
        sdk.assert_not_called()


def test_local_sdk_transport_does_not_follow_cloud_redirects(monkeypatch):
    from openai import APIStatusError

    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "local"}]})
        return httpx.Response(307, headers={"location": "https://cloud.invalid/v1/completions"})

    real_client = httpx.Client
    class RoutedClient(real_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("src.llm.reasoner.httpx.Client", RoutedClient)
    monkeypatch.setenv("LLAMA_API_KEY", "test-only-local-key")
    reasoner = HttpReasoner(base_url="http://127.0.0.1:8101/v1", destination="local")
    reasoner.load()
    with pytest.raises(APIStatusError) as error:
        reasoner.generate("private turn")
    assert error.value.status_code == 307
    assert seen == ["http://127.0.0.1:8101/v1/models", "http://127.0.0.1:8101/v1/completions"]
