"""Tests for MultiUserOrchestrator cleanup-mode routing (plan #2 OpenClaw)."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.orchestrator.context_persister import ContextPersister


def _make_orchestrator(persister=None):
    """Construct MUO with all heavy deps mocked."""
    return MultiUserOrchestrator(
        chroma_sync=MagicMock(),
        ha_client=MagicMock(),
        routine_manager=MagicMock(),
        persister=persister,
    )


async def _idle_processor():
    """Fake _process_queue that idles forever (cancelled on stop)."""
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        raise


async def _idle_async_loop():
    """Fake start_cleanup_loop_async that idles until cancelled."""
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        # mimic the real implementation which catches CancelledError to exit cleanly
        return


class TestCleanupModeRouting:
    @pytest.mark.asyncio
    async def test_start_calls_thread_cleanup_when_no_persister(self):
        """Without persister, start() must use start_cleanup_thread() (legacy)."""
        mgr = _make_orchestrator(persister=None)
        with patch.object(
            mgr.context_manager, "start_cleanup_thread"
        ) as mock_thread, patch.object(
            mgr.context_manager, "start_cleanup_loop_async"
        ) as mock_async, patch.object(
            mgr, "_process_queue", side_effect=_idle_processor
        ):
            await mgr.start()
            mock_thread.assert_called_once()
            mock_async.assert_not_called()
            assert mgr._async_cleanup_task is None
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_start_calls_async_cleanup_when_persister_set(self, tmp_path):
        """With persister, start() must use start_cleanup_loop_async()."""
        persister = ContextPersister(base_path=tmp_path / "ctx")
        mgr = _make_orchestrator(persister=persister)

        async_called = asyncio.Event()

        async def fake_async_loop():
            async_called.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                return

        with patch.object(
            mgr.context_manager, "start_cleanup_thread"
        ) as mock_thread, patch.object(
            mgr.context_manager, "start_cleanup_loop_async", side_effect=fake_async_loop
        ) as mock_async, patch.object(
            mgr, "_process_queue", side_effect=_idle_processor
        ):
            await mgr.start()
            await asyncio.wait_for(async_called.wait(), timeout=1.0)

            mock_async.assert_called_once()
            mock_thread.assert_not_called()
            assert mgr._async_cleanup_task is not None
            assert not mgr._async_cleanup_task.done()

            # Cleanup: cancel the cleanup task before stop() to avoid deadlock,
            # since we're patching stop_cleanup_loop_async would interfere.
            mgr._async_cleanup_task.cancel()
            try:
                await mgr._async_cleanup_task
            except asyncio.CancelledError:
                pass
            mgr._processor_task.cancel()
            try:
                await mgr._processor_task
            except asyncio.CancelledError:
                pass
            mgr._running = False

    @pytest.mark.asyncio
    async def test_stop_awaits_async_task_when_persister(self, tmp_path):
        """stop() with persister must signal stop_cleanup_loop_async + await the task."""
        persister = ContextPersister(base_path=tmp_path / "ctx")
        mgr = _make_orchestrator(persister=persister)

        # Track that stop was signalled. Also cancel the cleanup task so the
        # awaiting `await self._async_cleanup_task` in stop() unblocks
        # (the real stop_cleanup_loop_async sets _cleanup_running=False which
        # the real loop polls; here our fake loop only exits on cancel).
        stop_called = []

        def fake_stop():
            stop_called.append(True)
            if mgr._async_cleanup_task is not None:
                mgr._async_cleanup_task.cancel()

        with patch.object(
            mgr.context_manager, "stop_cleanup_loop_async", side_effect=fake_stop
        ), patch.object(
            mgr.context_manager, "start_cleanup_loop_async", side_effect=_idle_async_loop
        ), patch.object(
            mgr, "_process_queue", side_effect=_idle_processor
        ):
            await mgr.start()
            await asyncio.sleep(0.02)
            await mgr.stop()
            assert stop_called == [True]
