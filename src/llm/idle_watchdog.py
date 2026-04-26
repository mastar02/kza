"""
idle_watchdog: envuelve un async iterator y aborta si no llega chunk en N segundos.

Patrón en OpenClaw agent-loop.md (`agents.defaults.llm.idleTimeoutSeconds`).
Útil para 72B en CPU que puede colgarse silenciosamente sin cerrar el stream.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, TypeVar

from src.llm.error_classifier import IdleTimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def idle_watchdog(
    source: AsyncIterator[T],
    idle_seconds: float,
) -> AsyncIterator[T]:
    """
    Re-yield cada chunk de `source`. Si no llega nuevo chunk en `idle_seconds`,
    raise IdleTimeoutError.

    Args:
        source: async iterator/generator
        idle_seconds: segundos máximos sin chunk

    Yields:
        Cada chunk de `source`

    Raises:
        IdleTimeoutError: si el gap entre chunks excede `idle_seconds`
    """
    iterator = source.__aiter__()

    while True:
        try:
            chunk = await asyncio.wait_for(iterator.__anext__(), timeout=idle_seconds)
        except asyncio.TimeoutError as e:
            logger.warning(f"[IdleWatchdog] no chunks for {idle_seconds:.1f}s — aborting")
            raise IdleTimeoutError(idle_seconds) from e
        except StopAsyncIteration:
            return
        yield chunk
