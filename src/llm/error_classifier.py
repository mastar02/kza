"""
Mapper Exception → ErrorKind.

Patrón inspirado en OpenClaw model-failover.md ("What lands in the rate-limit /
timeout bucket"). Conservador: errores desconocidos se clasifican como PERMANENT
para no rotar erróneamente.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import ssl

import httpx

from src.llm.idle_watchdog import IdleTimeoutError
from src.llm.types import ErrorKind

logger = logging.getLogger(__name__)


# Re-export para backward compat con tests que importaban desde acá
__all__ = ["IdleTimeoutError", "classify_error"]


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

# Patrones transitorios de red/TLS/DNS/5xx (failover-worthy)
_TRANSIENT_PATTERNS = (
    "timed out",
    "timeout",
    "connection error",
    "connection reset",
    "temporarily unavailable",
    "name resolution",
    "getaddrinfo",
    "sslerror",
    "ssl error",
    "502",
    "503",
    "504",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
)


_WALK_LIMIT = 32


def is_consent_withdrawn(exc: BaseException) -> bool:
    """Whether a withdrawn cloud consent caused ``exc`` (bounded walk).

    Follows ``__cause__``/``__context__``, exception-group members and
    router attempt summaries (``attempts[].error_kind``), with a node budget
    that also terminates on cycles.
    """
    from src.core.egress import EgressDenied
    pending, seen = [exc], set()
    while pending and len(seen) < _WALK_LIMIT:
        item = pending.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, EgressDenied) and item.args == ("cloud_consent_withdrawn",):
            return True
        if any(getattr(attempt, "error_kind", None) is ErrorKind.CONSENT_WITHDRAWN
               for attempt in getattr(item, "attempts", None) or ()):
            return True
        if isinstance(item, BaseExceptionGroup):
            pending.extend(item.exceptions)
        pending.extend((item.__cause__, item.__context__))
    return False


def classify_error(exc: BaseException) -> ErrorKind:
    """Clasificar una exception en un ErrorKind."""

    # OQ-8: un retiro de consentimiento (aunque el SDK lo envuelva) es política:
    # se rota al siguiente candidato local sin cooldown.
    if is_consent_withdrawn(exc):
        return ErrorKind.CONSENT_WITHDRAWN

    # Idle timeout primero (es nuestro tipo, sabemos qué es)
    if isinstance(exc, IdleTimeoutError):
        return ErrorKind.IDLE_TIMEOUT

    # Timeouts a nivel asyncio o builtin
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return ErrorKind.TIMEOUT

    # Errores de red/TLS/DNS — siempre transitorio (failover-worthy)
    if isinstance(exc, (ssl.SSLError, socket.gaierror)):
        return ErrorKind.TIMEOUT

    # Native async HTTP errors may carry no text. Structured status also
    # prevents URL digits or provider prose from changing auth classification.
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in {401, 403}:
            return ErrorKind.AUTH
        if status == 429:
            return ErrorKind.RATE_LIMIT
        if status == 402:
            return ErrorKind.BILLING
        if status == 408 or status >= 500:
            return ErrorKind.TIMEOUT
        return ErrorKind.PERMANENT
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return ErrorKind.TIMEOUT

    # Texto del error para matchear patrones
    msg = str(exc).lower()

    if any(p in msg for p in _RATE_LIMIT_PATTERNS):
        return ErrorKind.RATE_LIMIT

    if any(p in msg for p in _BILLING_PATTERNS):
        return ErrorKind.BILLING

    # AUTH (401/403) ANTES de TRANSIENT: un error de auth nunca debe ser
    # failover-worthy, ni siquiera si el mensaje también contiene un patrón
    # transitorio (ej "401 Unauthorized: ssl error"). Auth-first hace que se
    # propague para acción humana en vez de rotar/cooldown silenciosamente.
    if any(p in msg for p in _AUTH_PATTERNS):
        return ErrorKind.AUTH

    # Patrones transitorios de red/TLS/DNS/5xx (failover-worthy)
    if any(p in msg for p in _TRANSIENT_PATTERNS):
        return ErrorKind.TIMEOUT

    if any(p in msg for p in _FORMAT_PATTERNS):
        return ErrorKind.FORMAT

    if isinstance(exc, ValueError):
        # ValueError sin patrón conocido = format
        return ErrorKind.FORMAT

    return ErrorKind.PERMANENT
