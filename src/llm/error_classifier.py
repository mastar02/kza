"""
Mapper Exception → ErrorKind.

Patrón inspirado en OpenClaw model-failover.md ("What lands in the rate-limit /
timeout bucket"). Conservador: errores desconocidos se clasifican como PERMANENT
para no rotar erróneamente.
"""

from __future__ import annotations

import asyncio
import logging

from src.llm.types import ErrorKind

logger = logging.getLogger(__name__)


class IdleTimeoutError(Exception):
    """Raised por idle_watchdog cuando el stream no emite chunks en N segundos."""

    def __init__(self, idle_seconds: float):
        self.idle_seconds = idle_seconds
        super().__init__(f"No chunks received for {idle_seconds:.1f}s")


# Patrones de texto que indican rate-limit / quota
_RATE_LIMIT_PATTERNS = (
    "429",
    "too many requests",
    "rate limit",
    "rate_limit",
    "throttl",  # Throttling, throttled
    "concurrency limit",
    "quota",
    "resource exhausted",
    "weekly limit",
    "monthly limit",
    "daily limit",
)

# Patrones que indican billing/credits agotados
_BILLING_PATTERNS = (
    "insufficient credits",
    "credit balance too low",
    "credits exhausted",
    "billing",
    "402",
)

# Patrones de auth (401/403)
_AUTH_PATTERNS = (
    "401",
    "unauthorized",
    "403",
    "forbidden",
    "invalid api key",
    "invalid token",
)

# Patrones de format error
_FORMAT_PATTERNS = (
    "invalid json",
    "json decode",
    "schema",
    "unexpected token",
)


def classify_error(exc: BaseException) -> ErrorKind:
    """Clasificar una exception en un ErrorKind."""

    # Idle timeout primero (es nuestro tipo, sabemos qué es)
    if isinstance(exc, IdleTimeoutError):
        return ErrorKind.IDLE_TIMEOUT

    # Timeouts a nivel asyncio o builtin
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return ErrorKind.TIMEOUT

    # Texto del error para matchear patrones
    msg = str(exc).lower()

    if any(p in msg for p in _RATE_LIMIT_PATTERNS):
        return ErrorKind.RATE_LIMIT

    if any(p in msg for p in _BILLING_PATTERNS):
        return ErrorKind.BILLING

    if any(p in msg for p in _AUTH_PATTERNS):
        return ErrorKind.AUTH

    if any(p in msg for p in _FORMAT_PATTERNS):
        return ErrorKind.FORMAT

    if isinstance(exc, ValueError):
        # ValueError sin patrón conocido = format
        return ErrorKind.FORMAT

    return ErrorKind.PERMANENT
