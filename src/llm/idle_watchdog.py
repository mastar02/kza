"""Idle watchdog para streams LLM.

Envuelve un async iterable y aborta si no llega un chunk en `idle_timeout_s`
segundos. Caso target: el 72B en CPU se cuelga durante un decode largo y nunca
emite el siguiente token. Sin watchdog, el caller queda esperando para siempre.

Patrón de OpenClaw `docs/concepts/agent-loop.md` adaptado a Python asyncio.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterable, AsyncIterator, Optional, TypeVar

T = TypeVar("T")


class IdleTimeoutError(Exception):
    """El stream no emitió un chunk dentro del idle_timeout_s configurado."""

    def __init__(self, idle_timeout_s: float, chunks_received: int = 0):
        super().__init__(
            f"stream idle por {idle_timeout_s}s tras {chunks_received} chunks"
        )
        self.idle_timeout_s = idle_timeout_s
        self.chunks_received = chunks_received


async def idle_watchdog(
    stream: AsyncIterable[T],
    idle_timeout_s: Optional[float],
) -> AsyncIterator[T]:
    """Yield chunks de `stream`; aborta si pasa más de `idle_timeout_s` sin chunk.

    Args:
        stream: Async iterable a envolver (típicamente un generador LLM).
        idle_timeout_s: Segundos máximos entre chunks. None o 0 = sin watchdog
            (pass-through directo).

    Yields:
        Cada chunk recibido del stream original.

    Raises:
        IdleTimeoutError: Si el stream no produce el siguiente chunk dentro del
            timeout. Las primeras chunks ya emitidas siguen siendo del caller.
    """
    if not idle_timeout_s:
        async for chunk in stream:
            yield chunk
        return

    iterator = stream.__aiter__()
    chunks_received = 0
    while True:
        try:
            chunk = await asyncio.wait_for(
                iterator.__anext__(),
                timeout=idle_timeout_s,
            )
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            raise IdleTimeoutError(
                idle_timeout_s=idle_timeout_s,
                chunks_received=chunks_received,
            )
        chunks_received += 1
        yield chunk
