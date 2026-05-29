from unittest.mock import MagicMock

import pytest

from src.llm.reasoner import HttpReasoner


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
async def test_complete_uses_chat_when_api_style_chat():
    r = HttpReasoner(base_url="https://api.minimax.io/v1", model="MiniMax-M2.7-highspeed", api_style="chat")
    r._client = _make_chat_client("la luz está encendida")
    r._resolved_model = "MiniMax-M2.7-highspeed"
    out = await r.complete("¿está la luz?", max_tokens=64)
    assert out == "la luz está encendida"
    _, kwargs = r._client.chat.completions.create.call_args
    assert kwargs["messages"] == [{"role": "user", "content": "¿está la luz?"}]
    assert kwargs["model"] == "MiniMax-M2.7-highspeed"


def test_call_chat_style_returns_completions_shaped_dict():
    r = HttpReasoner(base_url="https://api.minimax.io/v1", api_style="chat")
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
    r = HttpReasoner(base_url="http://127.0.0.1:8200/v1")  # default api_style="completions"
    r._client = client
    r._resolved_model = "local"
    result = r("hola")
    assert result["choices"][0]["text"] == "ok"
    _, kwargs = client.completions.create.call_args
    assert kwargs["prompt"] == "hola"
