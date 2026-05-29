"""Tests: strip de bloques de razonamiento <think>...</think> (MiniMax-M2.x)."""

from unittest.mock import MagicMock

from src.llm.reasoner import HttpReasoner, _strip_reasoning


def test_strip_complete_block():
    assert _strip_reasoning("<think>razono esto</think>\nLa luz está encendida") == "La luz está encendida"


def test_strip_no_tags_unchanged():
    # Modelos no-reasoning (GLM-Air/Qwen local) → no-op.
    assert _strip_reasoning("La luz está encendida") == "La luz está encendida"


def test_strip_multiple_blocks():
    assert _strip_reasoning("<think>a</think>uno<think>b</think>dos") == "unodos"


def test_strip_unclosed_think_returns_empty():
    # Reasoning truncado por max_tokens, sin respuesta todavía.
    assert _strip_reasoning("<think>pensando sin cerrar todavía") == ""


def test_strip_multiline_dotall():
    assert _strip_reasoning("<think>linea1\nlinea2\nlinea3</think>respuesta") == "respuesta"


def _chat_client_returning(text):
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.usage.prompt_tokens = 1
    resp.usage.completion_tokens = 2
    client.chat.completions.create.return_value = resp
    return client


def test_extract_text_strips_think():
    r = HttpReasoner(base_url="https://api.minimax.io/v1", api_style="chat")
    r._client = _chat_client_returning("<think>2+2 es 4</think>4")
    r._resolved_model = "MiniMax-M2.7"
    assert r("¿2+2?")["choices"][0]["text"] == "4"
    assert r.generate("¿2+2?") == "4"


def _stream_client(pieces):
    """Mock cuyo chat.completions.create(stream=True) devuelve chunks con delta.content."""
    chunks = []
    for p in pieces:
        ch = MagicMock()
        ch.choices = [MagicMock()]
        ch.choices[0].delta.content = p
        chunks.append(ch)
    client = MagicMock()
    client.chat.completions.create.return_value = iter(chunks)
    return client


def _collect(gen):
    toks = [c["token"] for c in gen]
    return "".join(toks)


def test_generate_stream_strips_think_first():
    r = HttpReasoner(base_url="https://api.minimax.io/v1", api_style="chat")
    r._resolved_model = "MiniMax-M2.7"
    # think completo en varios chunks, luego la respuesta
    r._client = _stream_client(["<think>", "razono ", "un poco", "</think>", "La ", "respuesta ", "final"])
    out = _collect(r.generate_stream("x"))
    assert "<think>" not in out and "</think>" not in out
    assert out == "La respuesta final"


def test_generate_stream_closing_tag_split_across_chunks():
    # El </think> partido entre chunks no debe filtrarse a la salida.
    r = HttpReasoner(base_url="https://api.minimax.io/v1", api_style="chat")
    r._resolved_model = "MiniMax-M2.7"
    r._client = _stream_client(["<think>r</th", "ink>", "ok"])
    out = _collect(r.generate_stream("x"))
    assert out == "ok"
    assert "think" not in out


def test_generate_stream_no_think_passthrough():
    r = HttpReasoner(base_url="https://api.minimax.io/v1", api_style="chat")
    r._resolved_model = "MiniMax-M2.7"
    r._client = _stream_client(["Hola", " mundo"])
    assert _collect(r.generate_stream("x")) == "Hola mundo"
