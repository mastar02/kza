"""Tests for idle_watchdog wrapping async streams."""

import asyncio
import pytest
from src.llm.idle_watchdog import idle_watchdog
from src.llm.error_classifier import IdleTimeoutError


async def _fast_stream():
    """Emite 3 chunks en ~30ms total."""
    for i in range(3):
        await asyncio.sleep(0.01)
        yield f"chunk-{i}"


async def _hanging_stream():
    """Emite 1 chunk y luego se cuelga."""
    yield "first"
    await asyncio.sleep(10.0)  # mucho más que el watchdog


async def _slow_stream(gap: float):
    """Emite 2 chunks separados por `gap` segundos."""
    yield "a"
    await asyncio.sleep(gap)
    yield "b"


class TestIdleWatchdog:
    @pytest.mark.asyncio
    async def test_passes_fast_stream(self):
        chunks = []
        async for chunk in idle_watchdog(_fast_stream(), idle_seconds=1.0):
            chunks.append(chunk)
        assert chunks == ["chunk-0", "chunk-1", "chunk-2"]

    @pytest.mark.asyncio
    async def test_aborts_hanging_stream(self):
        chunks = []
        with pytest.raises(IdleTimeoutError) as exc_info:
            async for chunk in idle_watchdog(_hanging_stream(), idle_seconds=0.1):
                chunks.append(chunk)
        assert chunks == ["first"]  # alcanzó a emitir 1 antes de colgarse
        assert exc_info.value.idle_seconds == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_passes_slow_stream_within_budget(self):
        # gap=0.05s, watchdog=0.2s → pasa
        chunks = []
        async for chunk in idle_watchdog(_slow_stream(0.05), idle_seconds=0.2):
            chunks.append(chunk)
        assert chunks == ["a", "b"]

    @pytest.mark.asyncio
    async def test_aborts_slow_stream_exceeding_budget(self):
        # gap=0.3s, watchdog=0.1s → aborta
        chunks = []
        with pytest.raises(IdleTimeoutError):
            async for chunk in idle_watchdog(_slow_stream(0.3), idle_seconds=0.1):
                chunks.append(chunk)
        assert chunks == ["a"]
