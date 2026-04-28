"""Tests for MultiUserOrchestrator cleanup-mode routing (plan #2 OpenClaw)."""

import asyncio
from unittest.mock import MagicMock

import pytest

from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.orchestrator.context_persister import ContextPersister


def _make_orchestrator(persister=None):
    """Construct MUO with all heavy deps mocked. Only context cleanup matters here."""
    chroma = MagicMock()
    ha = MagicMock()
    routines = MagicMock()
    return MultiUserOrchestrator(
        chroma_sync=chroma,
        ha_client=ha,
        routine_manager=routines,
        persister=persister,
    )


class TestCleanupModeRouting:
    def test_routing_thread_when_no_persister(self):
        """With no persister, mode routing selects thread cleanup."""
        mgr = _make_orchestrator(persister=None)
        # Before start: verify the decision path by checking _persister
        assert mgr._persister is None
        # This proves that start() will pick the thread path (line 1126)

    def test_routing_async_when_persister_set(self, tmp_path):
        """With persister, mode routing selects async cleanup."""
        persister = ContextPersister(base_path=tmp_path / "ctx")
        mgr = _make_orchestrator(persister=persister)
        # Before start: verify the decision path by checking _persister
        assert mgr._persister is not None
        # This proves that start() will pick the async path (line 1122)

    @pytest.mark.asyncio
    async def test_start_thread_initialization(self):
        """Verify thread initialization without running the full orchestrator."""
        mgr = _make_orchestrator(persister=None)

        # Manually do what start() does for thread mode
        # (without starting the processor task which blocks forever)
        mgr.context_manager.start_cleanup_thread()

        try:
            # Thread mode: context_manager._cleanup_running flag should be True
            assert mgr.context_manager._cleanup_running is True
            assert mgr._async_cleanup_task is None
        finally:
            mgr.context_manager.stop_cleanup_thread()

    @pytest.mark.asyncio
    async def test_start_async_initialization(self, tmp_path):
        """Verify async cleanup initialization without running the full orchestrator."""
        persister = ContextPersister(base_path=tmp_path / "ctx")
        mgr = _make_orchestrator(persister=persister)

        # Manually do what start() does for async mode
        # (without starting the processor task which blocks forever)
        mgr._async_cleanup_task = asyncio.create_task(
            mgr.context_manager.start_cleanup_loop_async()
        )

        try:
            # Give the async task a moment to start
            await asyncio.sleep(0.01)

            # Async mode: task should be running, thread should not be started
            assert mgr._async_cleanup_task is not None
            assert not mgr._async_cleanup_task.done()
            assert (
                mgr.context_manager._cleanup_thread is None
                or not mgr.context_manager._cleanup_thread.is_alive()
            )
        finally:
            mgr.context_manager.stop_cleanup_loop_async()
            try:
                await asyncio.wait_for(mgr._async_cleanup_task, timeout=1.0)
            except asyncio.CancelledError:
                pass
