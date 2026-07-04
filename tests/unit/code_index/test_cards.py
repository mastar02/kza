"""Tests del generador de cards (MiniMax vía gateway)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.code_index.cards import CardGenerator, _strip_think


def _fake_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_strip_think_removes_block():
    text = "<think>\nrazonando...\n</think>\n## Propósito\nAlgo."
    assert _strip_think(text).strip().startswith("## Propósito")


def test_strip_think_noop_without_block():
    assert _strip_think("## Propósito\nX") == "## Propósito\nX"


async def test_generate_returns_stripped_card():
    gen = CardGenerator(base_url="http://fake:8200/v1", model="test-model")
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=_fake_response("<think>x</think>## Propósito\nCard.")
                )
            )
        )
    )
    gen._client = fake_client

    card = await gen.generate("src/foo.py", "def f(): pass")

    assert card.startswith("## Propósito")
    call = fake_client.chat.completions.create.call_args
    assert call.kwargs["model"] == "test-model"
    prompt = call.kwargs["messages"][0]["content"]
    assert "src/foo.py" in prompt
    assert "def f(): pass" in prompt


async def test_generate_truncates_long_source():
    gen = CardGenerator(
        base_url="http://fake:8200/v1", model="m", max_source_chars=10
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=_fake_response("card"))
            )
        )
    )
    gen._client = fake_client

    await gen.generate("src/foo.py", "x" * 100)

    prompt = fake_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "x" * 11 not in prompt
