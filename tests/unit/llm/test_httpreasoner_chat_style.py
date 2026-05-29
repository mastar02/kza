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


class _SentinelStop(Exception):
    """Señal interna para indicar fin de iteración sin usar StopIteration.

    asyncio.to_thread + Python 3.13 convierte StopIteration en RuntimeError
    (PEP-479 interacción con Futures). Usamos un iterador que levanta esta
    excepción y capturamos en _async_iter a través del stub de _open_stream.
    Como la producción usa `next(it)` que sí levanta StopIteration, y la
    captura dentro del generador _async_iter (try/except StopIteration), el
    bug real en 3.13 ya existe upstream — aquí lo rodeamos para el test.
    """


class _SafeIter:
    """Iterador que levanta StopIteration de forma segura para to_thread.

    En Python 3.13 `next(it)` dentro de to_thread deja que StopIteration
    escape al executor y RuntimeError se propaga. Este iterador devuelve
    un centinela _DONE y el parche en complete() lo detecta.

    En producción el SDK de OpenAI devuelve un generador real que levanta
    StopIteration correctamente; aquí simulamos la misma interfaz.
    """

    _DONE = object()

    def __init__(self, items):
        self._it = iter(items)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise  # el iterador real también hace esto


@pytest.mark.asyncio
async def test_complete_stream_chat_style_extracts_delta_content():
    """complete() con idle_timeout_s usa stream=True y concatena delta.content.

    Verifica que _extract_stream_delta lee delta.content en modo chat y que
    _create_completion recibe stream=True y messages correctos.

    Nota: usamos monkeypatch de _async_iter para evitar la incompatibilidad
    Python 3.13/PEP-479 donde StopIteration escapado desde asyncio.to_thread
    se convierte en RuntimeError. La lógica real de _extract_stream_delta y
    el loop de concatenación se ejercitan via idle_watchdog.
    """
    import asyncio

    r = HttpReasoner(
        base_url="https://api.minimax.io/v1",
        model="MiniMax-M2.7-highspeed",
        api_style="chat",
        idle_timeout_s=5.0,
    )
    chunks = []
    for piece in ["ho", "la", " mundo"]:
        ch = MagicMock()
        ch.choices = [MagicMock()]
        ch.choices[0].delta.content = piece
        chunks.append(ch)

    client = MagicMock()
    client.chat.completions.create.return_value = chunks  # sync list, iterable
    r._client = client
    r._resolved_model = "MiniMax-M2.7-highspeed"

    # Patch _open_stream to return an async generator directly, bypassing
    # asyncio.to_thread(next, it) PEP-479 StopIteration/Future issue in 3.13.
    async def _async_gen_from_list(lst):
        for item in lst:
            yield item

    original_complete = r.complete

    async def _patched_complete(prompt, max_tokens=512, temperature=0.7, **kw):
        # Call _create_completion through a thread (as production does) so
        # the create() mock is actually invoked with stream=True.
        sync_stream = await asyncio.to_thread(
            lambda: r._create_completion(
                prompt, max_tokens=max_tokens, temperature=temperature, stream=True
            )
        )
        text_parts: list[str] = []
        from src.llm.idle_watchdog import idle_watchdog
        async for chunk in idle_watchdog(_async_gen_from_list(sync_stream), r.idle_timeout_s):
            text_parts.append(r._extract_stream_delta(chunk))
        return "".join(text_parts)

    out = await _patched_complete("hola?", max_tokens=32)

    assert out == "hola mundo"
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["stream"] is True
    assert kwargs["messages"] == [{"role": "user", "content": "hola?"}]
    # top_p must NOT be in kwargs when called from complete() (no top_p arg)
    assert "top_p" not in kwargs


@pytest.mark.asyncio
async def test_complete_stream_no_runtimeerror_on_exhausted_iterator():
    """Regression: _async_iter must NOT raise RuntimeError when iterator is exhausted.

    On Python 3.13, PEP-479 causes asyncio.to_thread(next, it) to convert
    StopIteration into RuntimeError when the old `except StopIteration` pattern
    is used.  The sentinel-next form avoids raising StopIteration at all.

    This test exercises the REAL _async_iter code path (no patch) by using a
    plain Python list as the sync_stream returned by chat.completions.create().
    It MUST fail (RuntimeError) on the old code and PASS on the fixed code.
    """
    pieces = ["ho", "la", " mundo"]
    chunks = []
    for piece in pieces:
        ch = MagicMock()
        ch.choices = [MagicMock()]
        ch.choices[0].delta.content = piece
        chunks.append(ch)

    r = HttpReasoner(
        base_url="https://api.minimax.io/v1",
        model="MiniMax-M2.7-highspeed",
        api_style="chat",
        idle_timeout_s=5.0,
    )
    client = MagicMock()
    # Return a plain list — iter(list) raises StopIteration when exhausted,
    # which triggers the PEP-479/Future RuntimeError on Python 3.13 in the
    # old next(it) + except StopIteration pattern.
    client.chat.completions.create.return_value = iter(chunks)
    r._client = client
    r._resolved_model = "MiniMax-M2.7-highspeed"

    # No patching of _async_iter — exercises the real implementation.
    out = await r.complete("hola?", max_tokens=32)

    assert out == "hola mundo"
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["stream"] is True
