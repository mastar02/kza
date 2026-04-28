"""Tests for context Compactor."""

import json
import pytest
from unittest.mock import AsyncMock

from src.orchestrator.compactor import (
    Compactor,
    CompactionResult,
    CompactionError,
    CompactionErrorKind,
)
from src.orchestrator.context_manager import ConversationTurn


def _turn(role: str, content: str, entities: list[str] | None = None) -> ConversationTurn:
    return ConversationTurn(role=role, content=content, entities=entities or [])


class TestCompactorHappyPath:
    @pytest.mark.asyncio
    async def test_returns_summary_from_json(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(
            return_value='{"summary": "El usuario controló la luz del escritorio."}'
        )
        compactor = Compactor(reasoner=reasoner)

        turns = [
            _turn("user", "Prendé la luz del escritorio", entities=["light.escritorio"]),
            _turn("assistant", "Listo"),
        ]
        result = await compactor.compact(turns, preserved_entities=["light.escritorio"])

        assert isinstance(result, CompactionResult)
        assert result.summary == "El usuario controló la luz del escritorio."
        assert result.preserved_ids == ("light.escritorio",)
        assert result.compacted_turns_count == 2
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_dedupes_preserved_entities(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(return_value='{"summary": "ok"}')
        compactor = Compactor(reasoner=reasoner)

        result = await compactor.compact(
            turns=[_turn("user", "x")],
            preserved_entities=["light.a", "light.a", "scene.b"],
        )
        assert result.preserved_ids == ("light.a", "scene.b")

    @pytest.mark.asyncio
    async def test_passes_max_tokens_to_reasoner(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(return_value='{"summary": "ok"}')
        compactor = Compactor(reasoner=reasoner, max_summary_tokens=128)

        await compactor.compact(turns=[_turn("user", "x")], preserved_entities=[])

        kwargs = reasoner.complete.await_args.kwargs
        assert kwargs.get("max_tokens") == 128


class TestCompactorErrorPaths:
    @pytest.mark.asyncio
    async def test_malformed_json_raises_parse_error(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(return_value="No JSON here, just text.")
        compactor = Compactor(reasoner=reasoner)

        with pytest.raises(CompactionError) as exc_info:
            await compactor.compact(turns=[_turn("user", "x")], preserved_entities=[])
        assert exc_info.value.kind == CompactionErrorKind.PARSE_FAILED

    @pytest.mark.asyncio
    async def test_extra_text_around_json_recovered(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(
            return_value='Pensemos... {"summary": "Hola"} fin.'
        )
        compactor = Compactor(reasoner=reasoner)

        result = await compactor.compact(
            turns=[_turn("user", "x")], preserved_entities=[]
        )
        assert result.summary == "Hola"

    @pytest.mark.asyncio
    async def test_empty_turns_raises(self):
        reasoner = AsyncMock()
        compactor = Compactor(reasoner=reasoner)

        with pytest.raises(CompactionError) as exc_info:
            await compactor.compact(turns=[], preserved_entities=[])
        assert exc_info.value.kind == CompactionErrorKind.EMPTY_INPUT

    @pytest.mark.asyncio
    async def test_timeout_wraps_into_compaction_error(self):
        import asyncio

        async def slow(*_, **__):
            await asyncio.sleep(10)
            return '{"summary": "no llega"}'

        reasoner = AsyncMock()
        reasoner.complete = slow
        compactor = Compactor(reasoner=reasoner, timeout_s=0.05)

        with pytest.raises(CompactionError) as exc_info:
            await compactor.compact(
                turns=[_turn("user", "x")], preserved_entities=[]
            )
        assert exc_info.value.kind == CompactionErrorKind.TIMEOUT
        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_reasoner_exception_wraps_into_compaction_error(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(side_effect=ConnectionError("boom"))
        compactor = Compactor(reasoner=reasoner)

        with pytest.raises(CompactionError) as exc_info:
            await compactor.compact(
                turns=[_turn("user", "x")], preserved_entities=[]
            )
        assert exc_info.value.kind == CompactionErrorKind.REASONER_FAILED


class TestCompactionResultInvariants:
    def test_negative_count_raises(self):
        with pytest.raises(ValueError):
            CompactionResult(
                summary="ok",
                preserved_ids=(),
                compacted_turns_count=0,
                model="m",
                latency_ms=1.0,
            )

    def test_negative_latency_raises(self):
        with pytest.raises(ValueError):
            CompactionResult(
                summary="ok",
                preserved_ids=(),
                compacted_turns_count=1,
                model="m",
                latency_ms=-1.0,
            )

    def test_empty_summary_raises(self):
        with pytest.raises(ValueError):
            CompactionResult(
                summary="   ",
                preserved_ids=(),
                compacted_turns_count=1,
                model="m",
                latency_ms=1.0,
            )
