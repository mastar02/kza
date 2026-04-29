"""Tests for execute_after_event — sync inline + async Task fire-and-forget."""

import asyncio
import logging
import pytest

from src.hooks.registry import HookRegistry
from src.hooks.runner import execute_after_event
from src.hooks.types import SttPayload


def _stt_payload() -> SttPayload:
    return SttPayload(
        timestamp=0.0, text="hola", latency_ms=100.0,
        user_id=None, zone_id=None, success=True,
    )


class TestSyncHandler:
    def test_sync_handler_runs_inline(self):
        reg = HookRegistry()
        ran = []

        def h(payload):
            ran.append(payload.text)

        reg.register_after("stt", h)
        execute_after_event(reg, "stt", _stt_payload())
        assert ran == ["hola"]

    def test_sync_handler_exception_logged(self, caplog):
        reg = HookRegistry()

        def h(payload):
            raise RuntimeError("sync boom")

        reg.register_after("stt", h)
        with caplog.at_level(logging.WARNING):
            execute_after_event(reg, "stt", _stt_payload())

        stats = reg.get_stats()
        assert stats["handler_failures"] == 1
        assert "sync boom" in (stats["handler_last_error"] or "")
        assert any("after-event handler" in rec.message.lower()
                   for rec in caplog.records)


class TestAsyncHandler:
    @pytest.mark.asyncio
    async def test_async_handler_runs_as_task(self):
        reg = HookRegistry()
        completed = asyncio.Event()
        seen = []

        async def h(payload):
            seen.append(payload.text)
            completed.set()

        reg.register_after("stt", h)
        execute_after_event(reg, "stt", _stt_payload())

        # Task scheduled but not yet awaited
        await asyncio.wait_for(completed.wait(), timeout=1.0)
        assert seen == ["hola"]

    @pytest.mark.asyncio
    async def test_async_strong_ref_in_after_tasks(self):
        reg = HookRegistry()
        gate = asyncio.Event()

        async def h(payload):
            await gate.wait()

        reg.register_after("stt", h)
        execute_after_event(reg, "stt", _stt_payload())

        # Task is in registry
        assert len(reg._after_tasks) == 1

        # Release and let task complete
        gate.set()
        await asyncio.sleep(0.02)
        # done_callback removes it
        assert len(reg._after_tasks) == 0

    @pytest.mark.asyncio
    async def test_async_exception_logged(self, caplog):
        reg = HookRegistry()
        completed = asyncio.Event()

        async def h(payload):
            try:
                raise RuntimeError("async boom")
            finally:
                completed.set()

        reg.register_after("stt", h)
        with caplog.at_level(logging.WARNING):
            execute_after_event(reg, "stt", _stt_payload())
            await asyncio.wait_for(completed.wait(), timeout=1.0)
            await asyncio.sleep(0.02)  # let done_callback fire

        stats = reg.get_stats()
        assert stats["handler_failures"] == 1
        assert "async boom" in (stats["handler_last_error"] or "")


class TestNoEventLoop:
    def test_async_handler_no_loop_logs_warning(self, caplog):
        """If no event loop is running, async handlers are skipped with a warning."""
        reg = HookRegistry()

        async def h(payload):
            pass

        reg.register_after("stt", h)
        with caplog.at_level(logging.WARNING):
            execute_after_event(reg, "stt", _stt_payload())

        # No crash; warning logged
        assert any("no event loop" in rec.message.lower() for rec in caplog.records)
