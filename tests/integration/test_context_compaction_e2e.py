"""E2E: trigger → compaction → snapshot → hydration."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.orchestrator.context_manager import ContextManager
from src.orchestrator.compactor import Compactor, CompactionResult
from src.orchestrator.context_persister import ContextPersister


@pytest.mark.asyncio
async def test_full_lifecycle(tmp_path: Path):
    # Mock reasoner that returns canonical JSON
    reasoner = AsyncMock()
    reasoner.complete = AsyncMock(
        return_value='{"summary": "El usuario controló iluminación en la oficina."}'
    )
    reasoner._resolved_model = "qwen3-30b-a3b"

    compactor = Compactor(reasoner=reasoner, max_summary_tokens=200, timeout_s=30.0)
    persister = ContextPersister(base_path=tmp_path / "contexts")

    mgr = ContextManager(
        max_history=20,
        inactive_timeout=0.05,  # expires quickly for snapshot
        cleanup_interval=0.05,
        compactor=compactor,
        persister=persister,
        compaction_threshold=6,
        keep_recent_turns=3,
    )

    # === Session 1 ===
    mgr.get_or_create("alice", "Alice")
    for i in range(7):
        mgr.add_turn("alice", "user", f"comando {i}", entities=[f"light.{i}"])

    # Wait for in-memory compaction
    await asyncio.sleep(0.1)

    ctx_pre_snapshot = mgr.get("alice")
    assert ctx_pre_snapshot.compacted_summary is not None
    assert "iluminación" in ctx_pre_snapshot.compacted_summary
    # 7 turns: compaction_threshold=6 triggers at turn 6 → compactor compacts 4 (7-keep_recent_turns)
    # → remaining: 7 - 4 = 3 turns (keep_recent_turns)
    assert len(ctx_pre_snapshot.conversation_history) == 3

    # === Cleanup loop async — triggers snapshot ===
    cleanup_task = asyncio.create_task(mgr.start_cleanup_loop_async())
    await asyncio.sleep(0.3)  # > inactive_timeout + cleanup_interval
    mgr.stop_cleanup_loop_async()
    await cleanup_task

    assert persister.exists("alice")
    assert mgr.get("alice") is None  # purged from memory

    # === Session 2 — hydration ===
    ctx_hydrated = mgr.get_or_create("alice", "Alice")
    assert ctx_hydrated.compacted_summary is not None
    assert "iluminación" in ctx_hydrated.compacted_summary
    assert ctx_hydrated.session_count == 2
    assert ctx_hydrated.conversation_history == []
    # preserved_ids preserved literally
    assert any(eid.startswith("light.") for eid in ctx_hydrated.preserved_ids)
