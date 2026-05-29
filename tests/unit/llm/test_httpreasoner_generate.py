"""Tests for HttpReasoner.generate / generate_stream drop-in interface.

These methods are consumed by MultiUserOrchestrator._process_llm_request (slow path),
which checks hasattr(self.llm, 'generate_stream') and falls back to .generate().
"""

from unittest.mock import MagicMock

from src.llm.reasoner import HttpReasoner


def _chat_client(text):
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.usage.prompt_tokens = 3
    resp.usage.completion_tokens = 4
    client.chat.completions.create.return_value = resp
    return client


def test_generate_returns_text():
    r = HttpReasoner(base_url="https://api.minimax.io/v1", api_style="chat")
    r._client = _chat_client("hola mundo")
    r._resolved_model = "MiniMax-M2.7-highspeed"
    assert r.generate("hi") == "hola mundo"


def test_httpreasoner_has_drop_in_interface():
    # _process_llm_request does hasattr(self.llm, 'generate_stream') / .generate
    assert hasattr(HttpReasoner, "generate")
    assert hasattr(HttpReasoner, "generate_stream")


def test_generate_stream_yields_token_dicts():
    r = HttpReasoner(base_url="https://api.minimax.io/v1", api_style="chat")
    chunks = []
    for piece in ["ho", "la"]:
        ch = MagicMock()
        ch.choices = [MagicMock()]
        ch.choices[0].delta.content = piece
        chunks.append(ch)
    client = MagicMock()
    client.chat.completions.create.return_value = iter(chunks)
    r._client = client
    r._resolved_model = "MiniMax-M2.7-highspeed"

    out = list(r.generate_stream("hi"))
    assert [c["token"] for c in out] == ["ho", "la"]
    assert out[-1]["text"] == "hola"
    assert out[-1]["token_count"] == 2
    # confirm it streamed via chat with stream=True
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["stream"] is True
