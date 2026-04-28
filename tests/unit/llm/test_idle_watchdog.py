"""Tests for idle watchdog wrapper."""

import asyncio

import pytest

from src.llm.idle_watchdog import IdleTimeoutError, idle_watchdog


@pytest.mark.asyncio
class TestIdleWatchdogHappyPath:
    async def test_passes_chunks_when_arriving_in_time(self):
        async def stream():
            for c in ["a", "b", "c"]:
                await asyncio.sleep(0.01)
                yield c

        out = []
        async for chunk in idle_watchdog(stream(), idle_timeout_s=0.5):
            out.append(chunk)
        assert out == ["a", "b", "c"]

    async def test_finite_stream_returns_cleanly(self):
        async def stream():
            yield "only"

        out = [c async for c in idle_watchdog(stream(), idle_timeout_s=1.0)]
        assert out == ["only"]

    async def test_empty_stream_returns_no_chunks(self):
        async def stream():
            return
            yield  # unreachable

        out = [c async for c in idle_watchdog(stream(), idle_timeout_s=1.0)]
        assert out == []


@pytest.mark.asyncio
class TestIdleWatchdogTimeout:
    async def test_raises_when_stream_stalls(self):
        async def stream():
            yield "first"
            await asyncio.sleep(2.0)  # stall longer than timeout
            yield "never_arrives"

        out = []
        with pytest.raises(IdleTimeoutError) as exc_info:
            async for chunk in idle_watchdog(stream(), idle_timeout_s=0.1):
                out.append(chunk)
        assert out == ["first"]
        assert exc_info.value.idle_timeout_s == 0.1

    async def test_raises_on_initial_stall(self):
        async def stream():
            await asyncio.sleep(2.0)
            yield "never"

        with pytest.raises(IdleTimeoutError):
            async for _ in idle_watchdog(stream(), idle_timeout_s=0.1):
                pass

    async def test_zero_timeout_disabled(self):
        """idle_timeout_s=0 or None → watchdog acts as pass-through."""
        async def stream():
            await asyncio.sleep(0.05)
            yield "ok"

        out = [c async for c in idle_watchdog(stream(), idle_timeout_s=None)]
        assert out == ["ok"]
