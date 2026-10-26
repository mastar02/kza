import json
from unittest.mock import MagicMock

import pytest

from src.llm.reasoner import HttpReasoner
from tests.mocks.llm_http import event
from tests.unit.llm.test_http_async_stream import setup


def _make_chat_client(reply_text):
    """Mock de cliente OpenAI que solo soporta chat.completions (como MiniMax)."""
    client = MagicMock()
    msg = MagicMock()
    msg.message.content = reply_text
    resp = MagicMock()
    resp.choices = [msg]
    resp.usage.prompt_tokens = 5
    resp.usage.completion_tokens = 7
    client.chat.completions.create.return_value = resp
    client.completions.create.side_effect = AttributeError("no legacy completions")
    return client


@pytest.mark.asyncio
async def test_complete_uses_chat_when_api_style_chat(monkeypatch):
    r, endpoint = setup(monkeypatch)
    endpoint.completion_payload = {
        "choices": [{"message": {"content": "la luz está encendida"}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 7},
    }
    out = await r.complete("¿está la luz?", max_tokens=64)
    assert out == "la luz está encendida"
    kwargs = json.loads(endpoint.requests[-1].content)
    assert kwargs["messages"] == [{"role": "user", "content": "¿está la luz?"}]
    assert kwargs["model"] == "configured-model"
    assert kwargs["stream"] is False
    assert r._last_metrics["tokens"] == 7
    assert endpoint.client.is_closed()


def test_call_chat_style_returns_completions_shaped_dict():
    r = HttpReasoner(base_url="https://api.minimax.io/v1", api_style="chat", cloud_consent=True)
    r._client = _make_chat_client("hola")
    r._resolved_model = "MiniMax-M2.7-highspeed"
    result = r("dummy prompt")
    assert result["choices"][0]["text"] == "hola"
    assert result["usage"]["completion_tokens"] == 7


def test_completions_style_unchanged_legacy():
    """api_style por defecto ('completions') sigue usando completions.create con prompt=."""
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(text="ok")]
    resp.usage.prompt_tokens = 1
    resp.usage.completion_tokens = 2
    client.completions.create.return_value = resp
    r = HttpReasoner(base_url="http://127.0.0.1:8200/v1", cloud_consent=True)  # default api_style="completions"
    r._client = client
    r._resolved_model = "local"
    result = r("hola")
    assert result["choices"][0]["text"] == "ok"
    _, kwargs = client.completions.create.call_args
    assert kwargs["prompt"] == "hola"



@pytest.mark.asyncio
async def test_complete_stream_chat_style_extracts_delta_content(monkeypatch):
    r, endpoint = setup(monkeypatch, [event("ho"), event("la"), event(" mundo"),
                                     event(finish="stop")], idle_timeout_s=5.0)
    out = await r.complete("hola?", max_tokens=32)
    assert out == "hola mundo"
    kwargs = json.loads(endpoint.requests[-1].content)
    assert kwargs["stream"] is True
    assert kwargs["messages"] == [{"role": "user", "content": "hola?"}]
    assert "top_p" not in kwargs
    assert endpoint.body.closed and endpoint.client.is_closed()


@pytest.mark.asyncio
async def test_complete_stream_no_runtimeerror_on_exhausted_iterator(monkeypatch):
    """Real native EOF closes both resources without a sync iterator bridge."""
    r, endpoint = setup(monkeypatch, [event("hola mundo", finish="stop")], idle_timeout_s=5.0)
    assert await r.complete("hola?", max_tokens=32) == "hola mundo"
    assert endpoint.body.closed and endpoint.client.is_closed()
