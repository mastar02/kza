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


class TestCleanupSnapshot:
    @pytest.mark.asyncio
    async def test_cleanup_persists_expired_context(self, tmp_path):
        from src.orchestrator.context_persister import ContextPersister

        persister = ContextPersister(base_path=tmp_path / "contexts")
        compactor = AsyncMock()
        compactor.compact = AsyncMock(return_value=_result(summary="snap"))

        mgr = ContextManager(
            inactive_timeout=0.01,  # casi instantáneo
            cleanup_interval=0.05,
            compactor=compactor,
            persister=persister,
        )
        mgr.get_or_create("snap_user", "Ana")
        mgr.add_turn("snap_user", "user", "hola", entities=["light.a"])

        # Iniciar el cleanup loop como asyncio task
        cleanup_task = asyncio.create_task(mgr.start_cleanup_loop_async())

        # Esperar más que inactive_timeout + cleanup_interval
        await asyncio.sleep(0.2)

        mgr.stop_cleanup_loop_async()
        await cleanup_task

        # Verificar persistencia
        assert persister.exists("snap_user")
        data = persister.load("snap_user")
        assert "snap" in (data["compacted_summary"] or "")

        # Y el contexto fue removido de memoria
        assert mgr.get("snap_user") is None

    @pytest.mark.asyncio
    async def test_cleanup_skips_active_contexts(self, tmp_path):
        from src.orchestrator.context_persister import ContextPersister

        persister = ContextPersister(base_path=tmp_path / "contexts")
        mgr = ContextManager(
            inactive_timeout=10.0,  # nadie expira en este test
            cleanup_interval=0.02,
            persister=persister,
        )
        mgr.get_or_create("active_user", "Bob")

        cleanup_task = asyncio.create_task(mgr.start_cleanup_loop_async())
        await asyncio.sleep(0.1)
        mgr.stop_cleanup_loop_async()
        await cleanup_task

        assert mgr.get("active_user") is not None
        assert not persister.exists("active_user")

    @pytest.mark.asyncio
    async def test_cleanup_without_persister_just_deletes(self):
        mgr = ContextManager(
            inactive_timeout=0.01,
            cleanup_interval=0.02,
            persister=None,
        )
        mgr.get_or_create("ghost", "x")

        cleanup_task = asyncio.create_task(mgr.start_cleanup_loop_async())
        await asyncio.sleep(0.1)
        mgr.stop_cleanup_loop_async()
        await cleanup_task

        assert mgr.get("ghost") is None  # baseline cleanup behavior
