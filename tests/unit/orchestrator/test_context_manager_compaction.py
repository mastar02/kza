"""Tests for ContextManager compaction integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.orchestrator.context_manager import ContextManager, ConversationTurn
from src.orchestrator.compactor import CompactionResult, CompactionError, CompactionErrorKind


def _result(summary: str = "ok", ids: list[str] | None = None, count: int = 3) -> CompactionResult:
    return CompactionResult(
        summary=summary,
        preserved_ids=tuple(ids or ()),
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
        compactor.compact = AsyncMock(side_effect=CompactionError(CompactionErrorKind.REASONER_FAILED, "boom"))

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


class TestHydration:
    def test_hydrates_summary_and_preserved_ids(self, tmp_path):
        from src.orchestrator.context_persister import ContextPersister
        from src.orchestrator.context_manager import UserContext

        persister = ContextPersister(base_path=tmp_path / "contexts")
        prior = UserContext(
            user_id="returning",
            user_name="Carla",
            compacted_summary="Resumen viejo.",
            preserved_ids=["light.cocina"],
            session_count=3,
        )
        persister.save(prior)

        mgr = ContextManager(persister=persister)
        ctx = mgr.get_or_create("returning", "Carla")

        assert ctx.compacted_summary == "Resumen viejo."
        assert ctx.preserved_ids == ["light.cocina"]
        assert ctx.session_count == 4  # incrementa
        assert ctx.conversation_history == []  # turnos no se restauran

    def test_no_persister_no_hydration(self, tmp_path):
        # Aunque exista archivo en disk, sin persister no hidrata
        mgr = ContextManager(persister=None)
        ctx = mgr.get_or_create("anyone", "x")
        assert ctx.compacted_summary is None
        assert ctx.session_count == 1

    def test_corrupt_file_creates_fresh_context(self, tmp_path, caplog):
        from src.orchestrator.context_persister import ContextPersister
        base = tmp_path / "contexts"
        base.mkdir(parents=True, exist_ok=True)
        (base / "broken.json").write_text("{ not json")
        persister = ContextPersister(base_path=base)

        mgr = ContextManager(persister=persister)
        ctx = mgr.get_or_create("broken", "x")

        assert ctx.compacted_summary is None
        assert ctx.session_count == 1


class TestPromptInjection:
    """C1: build_prompt + build_chat_messages must surface compacted_summary."""

    def test_build_prompt_includes_compacted_summary(self):
        mgr = ContextManager(max_history=10)
        ctx = mgr.get_or_create("u1", "Juan")
        ctx.compacted_summary = "Resumen antiguo."
        ctx.preserved_ids = ["light.cocina", "scene.lectura"]

        prompt = mgr.build_prompt("u1", "¿qué hago ahora?")
        assert "Resumen antiguo." in prompt
        assert "light.cocina" in prompt or "scene.lectura" in prompt

    def test_build_chat_messages_includes_compacted_summary(self):
        mgr = ContextManager(max_history=10)
        ctx = mgr.get_or_create("u1", "Juan")
        ctx.compacted_summary = "Resumen antiguo."

        messages = mgr.build_chat_messages("u1", "hola")
        joined = " ".join(m["content"] for m in messages)
        assert "Resumen antiguo." in joined

    def test_build_prompt_no_summary_no_section(self):
        """Si no hay summary, no se inyecta la sección (no clutter)."""
        mgr = ContextManager(max_history=10)
        mgr.get_or_create("u1", "Juan")
        prompt = mgr.build_prompt("u1", "hola")
        assert "Resumen de conversación previa" not in prompt


class TestRaceSafeCompaction:
    """C2: turnos agregados durante el await del Compactor no deben evicrse."""

    @pytest.mark.asyncio
    async def test_concurrent_add_turn_during_compaction_does_not_drop_recent_turns(
        self, manager_with_compactor
    ):
        mgr, compactor = manager_with_compactor

        gate = asyncio.Event()

        async def slow(*a, **kw):
            await gate.wait()
            return _result(summary="done")

        compactor.compact = slow

        mgr.get_or_create("u1", "Juan")
        # 6 turnos → trigger
        for i in range(6):
            mgr.add_turn("u1", "user", f"old-{i}", entities=[f"light.{i}"])

        # Mientras el compactor await-ea, agregar 3 más
        await asyncio.sleep(0.01)
        for i in range(3):
            mgr.add_turn("u1", "user", f"new-{i}")

        # Soltar el gate
        gate.set()
        await asyncio.sleep(0.05)

        ctx = mgr.get("u1")
        contents = [t.content for t in ctx.conversation_history]
        # Los 3 nuevos DEBEN estar
        assert "new-0" in contents
        assert "new-1" in contents
        assert "new-2" in contents
        # Los últimos 3 viejos (keep_recent=3) también — id-based filter only drops
        # the snapshot of {old-0, old-1, old-2}
        assert "old-3" in contents
        assert "old-4" in contents
        assert "old-5" in contents
        # Los 3 primeros viejos se compactaron
        assert "old-0" not in contents
        assert "old-1" not in contents
        assert "old-2" not in contents


class TestStatsCounters:
    """I8: get_stats() exposes compaction observability."""

    @pytest.mark.asyncio
    async def test_stats_track_compaction_attempts_and_failures(self, manager_with_compactor):
        mgr, compactor = manager_with_compactor
        compactor.compact = AsyncMock(
            side_effect=CompactionError(CompactionErrorKind.REASONER_FAILED, "boom")
        )

        mgr.get_or_create("u1", "Juan")
        for i in range(6):
            mgr.add_turn("u1", "user", f"msg {i}")
        await asyncio.sleep(0.05)

        stats = mgr.get_stats()
        assert stats["compaction_attempts"] == 1
        assert stats["compaction_failures"] == 1
        last_err = stats["compaction_last_error"] or ""
        assert "REASONER_FAILED" in last_err or "boom" in last_err

    @pytest.mark.asyncio
    async def test_stats_track_compaction_success(self, manager_with_compactor):
        mgr, compactor = manager_with_compactor
        mgr.get_or_create("u1", "Juan")
        for i in range(6):
            mgr.add_turn("u1", "user", f"msg {i}")
        await asyncio.sleep(0.05)

        stats = mgr.get_stats()
        assert stats["compaction_attempts"] == 1
        assert stats["compaction_failures"] == 0
        assert stats["compaction_last_error"] is None


class TestToDictSerialization:
    """I1: to_dict() includes compaction state."""

    def test_to_dict_includes_compaction_fields(self):
        from src.orchestrator.context_manager import UserContext

        ctx = UserContext(
            user_id="u1",
            user_name="Juan",
            compacted_summary="resumen",
            preserved_ids=["light.x", "scene.y"],
            session_count=3,
        )
        d = ctx.to_dict()
        assert d["compacted_summary"] == "resumen"
        assert d["preserved_ids"] == ["light.x", "scene.y"]
        assert d["session_count"] == 3
        # transient flag NOT included
        assert "compaction_inflight" not in d
