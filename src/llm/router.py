"""
LLMRouter: candidate chain con cooldown skip y failover.

Patrón en OpenClaw model-failover.md "Runtime flow":
1. Resolve session state (skip — no session aquí)
2. Build candidate chain
3. Try current provider
4. Advance on failover-worthy errors
5. Persist fallback override (skip — sin sesión)
6. Roll back narrowly (skip)
7. Throw FallbackSummaryError if exhausted

Nota: el router NO maneja retry attempts dentro del mismo endpoint —
eso es responsabilidad del cliente (HttpReasoner / FastRouter ya tienen su
propio retry interno). Aquí solo rotamos entre endpoints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.llm.cooldown import CooldownManager
from src.llm.error_classifier import classify_error
from src.llm.types import ErrorKind, LLMEndpoint, RouterResult

logger = logging.getLogger(__name__)


@dataclass
class FailedAttempt:
    """Registro per-attempt para el FallbackSummaryError."""
    endpoint_id: str
    error_kind: ErrorKind
    error_message: str


class FallbackSummaryError(Exception):
    """Todos los candidatos fallaron o están en cooldown."""

    def __init__(
        self,
        attempts: list[FailedAttempt],
        soonest_retry_at: Optional[float] = None,
    ):
        self.attempts = attempts
        self.soonest_retry_at = soonest_retry_at
        retry_str = (
            f" (next attempt at epoch {soonest_retry_at:.0f})"
            if soonest_retry_at else ""
        )
        super().__init__(
            f"All {len(attempts)} LLM endpoints exhausted{retry_str}: "
            + "; ".join(f"{a.endpoint_id}={a.error_kind.value}" for a in attempts)
        )


class LLMRouter:
    """Router con candidate chain ordenada por priority."""

    def __init__(
        self,
        endpoints: list[LLMEndpoint],
        cooldown_manager: CooldownManager,
        metrics_tracker=None,
    ):
        if not endpoints:
            raise ValueError("LLMRouter requires at least one endpoint")
        # Orden estable por priority asc
        self._endpoints = sorted(endpoints, key=lambda e: e.priority)
        self._cd = cooldown_manager
        self._metrics = metrics_tracker

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        **kwargs,
    ) -> RouterResult:
        """
        Iterar candidatos en orden de priority. Saltar los en cooldown.
        Ante fallo failover-worthy → cooldown + next. Ante AUTH/PERMANENT → propagar.
        """
        start = time.perf_counter()
        attempts: list[FailedAttempt] = []
        soonest_retry: Optional[float] = None

        for ep in self._endpoints:
            if not self._cd.is_available(ep.id):
                next_at = self._cd.next_attempt_at(ep.id)
                if soonest_retry is None or next_at < soonest_retry:
                    soonest_retry = next_at
                logger.debug(f"[LLMRouter] skip {ep.id} (cooldown until {next_at:.0f})")
                continue

            try:
                logger.debug(f"[LLMRouter] try {ep.id} (kind={ep.kind.value})")
                text = await ep.client.complete(prompt, max_tokens=max_tokens, **kwargs)
            except Exception as exc:
                kind = classify_error(exc)
                logger.warning(
                    f"[LLMRouter] {ep.id} failed: kind={kind.value} "
                    f"err={type(exc).__name__}: {exc}"
                )

                if not kind.is_failover_worthy():
                    # AUTH/PERMANENT — no rotamos. Propagamos.
                    raise

                self._cd.record_failure(ep.id, kind)
                attempts.append(FailedAttempt(
                    endpoint_id=ep.id,
                    error_kind=kind,
                    error_message=str(exc),
                ))
                continue

            # Éxito
            self._cd.record_success(ep.id)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Sample metrics si el cliente las dejó en _last_metrics tras la call.
            if self._metrics is not None:
                lm = getattr(ep.client, "_last_metrics", None)
                if lm and lm.get("tokens", 0) > 0:
                    self._metrics.record(ep.id, lm["tokens"], lm["ms"])

            return RouterResult(
                text=text,
                endpoint_id=ep.id,
                attempts=len(attempts) + 1,
                elapsed_ms=elapsed_ms,
            )

        # Todos los endpoints fallaron o están en cooldown
        raise FallbackSummaryError(attempts=attempts, soonest_retry_at=soonest_retry)
