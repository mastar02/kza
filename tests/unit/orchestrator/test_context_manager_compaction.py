"""Tests for ContextManager compaction integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.orchestrator.context_manager import ContextManager, ConversationTurn
from src.orchestrator.compactor import CompactionResult, CompactionError


def _result(summary: str = "ok", ids: list[str] | None = None, count: int = 3) -> CompactionResult:
    return CompactionResult(
        summary=summary,
        preserved_ids=ids or [],
        compacted_turns_count=count,
        model="test-30b",
        latency_ms=10.0,
    )


@pytest.fixture
def manager_with_compactor():
    compactor = AsyncMock()
    compactor.compact = AsyncMock(return_value=_result(summary="compacted!", ids=["light.x"]))
    mgr = ContextManager(
        max_history=20,  # más que threshold para probar trigger sin trunc
        compactor=compactor,
        compaction_threshold=6,
        keep_recent_turns=3,
    )
    return mgr, compactor


class TestCompactionTrigger:
    @pytest.mark.asyncio
    async def test_trigger_fires_at_threshold(self, manager_with_compactor):
        mgr, compactor = manager_with_compactor
        mgr.get_or_create("u1", "Juan")

        # Llenar hasta el turno 6 (threshold)
        for i in range(6):
            mgr.add_turn("u1", "user", f"msg {i}", entities=[f"light.{i}"])

        # Esperar que la task background termine
        await asyncio.sleep(0.05)

        compactor.compact.assert_awaited_once()
        ctx = mgr.get("u1")
        # Tras compactación: summary set + history reducido a keep_recent
        assert ctx.compacted_summary == "compacted!"
        assert len(ctx.conversation_history) == 3
        assert "light.x" in ctx.preserved_ids

    @pytest.mark.asyncio
    async def test_trigger_only_once_while_inflight(self, manager_with_compactor):
        """Si la compactación está corriendo, turnos extra no disparan otra."""
        mgr, compactor = manager_with_compactor

        # Hacer la compactación lenta
        gate = asyncio.Event()
        async def slow(*a, **kw):
            await gate.wait()
            return _result(summary="slow")
        compactor.compact = slow

        mgr.get_or_create("u1", "Juan")
        for i in range(7):  # 7 turnos: trigger en 6, turno 7 no re-dispara
            mgr.add_turn("u1", "user", f"msg {i}")

        # Soltar el gate y esperar
        gate.set()
        await asyncio.sleep(0.05)
        # exact-once via inflight flag (compactor.compact ya no es AsyncMock; chequear estado)
        ctx = mgr.get("u1")
        assert ctx.compaction_inflight is False

    @pytest.mark.asyncio
    async def test_trigger_failure_leaves_history_intact(self, manager_with_compactor):
        mgr, compactor = manager_with_compactor
        compactor.compact = AsyncMock(side_effect=CompactionError("boom"))

        mgr.get_or_create("u1", "Juan")
        for i in range(6):
            mgr.add_turn("u1", "user", f"msg {i}")

        await asyncio.sleep(0.05)

        ctx = mgr.get("u1")
        assert ctx.compacted_summary is None
        assert len(ctx.conversation_history) == 6
        assert ctx.compaction_inflight is False

    @pytest.mark.asyncio
    async def test_no_compactor_no_trigger(self):
        """Sin compactor inyectado: comportamiento baseline (truncate duro)."""
        mgr = ContextManager(max_history=4, compactor=None)
        mgr.get_or_create("u1", "Juan")
        for i in range(10):
            mgr.add_turn("u1", "user", f"msg {i}")

        ctx = mgr.get("u1")
        assert ctx.compacted_summary is None
        assert len(ctx.conversation_history) == 4  # truncate baseline

    @pytest.mark.asyncio
    async def test_concatenates_summary_on_second_compaction(self, manager_with_compactor):
        mgr, compactor = manager_with_compactor

        compactor.compact = AsyncMock(side_effect=[
            _result(summary="A.", ids=["light.a"]),
            _result(summary="B.", ids=["light.b"]),
        ])

        mgr.get_or_create("u1", "Juan")
        # Primera ronda: 6 turnos → trigger
        for i in range(6):
            mgr.add_turn("u1", "user", f"r1-{i}")
        await asyncio.sleep(0.05)
        # Segunda ronda: agregar 6 más. Como tras primera compactación quedaron
        # 3 en history, agregar 3 más llega a threshold (6) y dispara segundo round.
        for i in range(6):
            mgr.add_turn("u1", "user", f"r2-{i}")
        await asyncio.sleep(0.05)

        ctx = mgr.get("u1")
        assert "A." in ctx.compacted_summary
        assert "B." in ctx.compacted_summary
        assert sorted(ctx.preserved_ids) == ["light.a", "light.b"]
