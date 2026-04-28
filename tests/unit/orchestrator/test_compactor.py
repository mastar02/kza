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
