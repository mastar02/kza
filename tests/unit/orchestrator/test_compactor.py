"""Tests for context Compactor."""

import json
import pytest
from unittest.mock import AsyncMock

from src.orchestrator.compactor import (
    Compactor,
    CompactionResult,
    CompactionError,
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
        assert result.preserved_ids == ["light.escritorio"]
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
        assert sorted(result.preserved_ids) == ["light.a", "scene.b"]

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
    async def test_malformed_json_falls_back_to_text(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(return_value="No JSON here, just text.")
        compactor = Compactor(reasoner=reasoner)

        result = await compactor.compact(
            turns=[_turn("user", "x")], preserved_entities=[]
        )
        assert "No JSON here" in result.summary

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

        with pytest.raises(CompactionError):
            await compactor.compact(turns=[], preserved_entities=[])

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
        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_reasoner_exception_wraps_into_compaction_error(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(side_effect=ConnectionError("boom"))
        compactor = Compactor(reasoner=reasoner)

        with pytest.raises(CompactionError):
            await compactor.compact(
                turns=[_turn("user", "x")], preserved_entities=[]
            )
