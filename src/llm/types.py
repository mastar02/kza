"""
LLM Router type primitives.

Patrón inspirado en OpenClaw model-failover (docs/concepts/model-failover.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EndpointKind(Enum):
    """Tipo de endpoint LLM. Usado para logging y selección de cliente."""

    FAST_ROUTER = "fast_router"        # vLLM 7B :8100 (compartido infra)
    HTTP_REASONER = "http_reasoner"    # llama-cpp 72B :8200 (kza-72b.service)
    LOCAL_REASONER = "local_reasoner"  # llama-cpp en proceso (legacy)
    CLOUD = "cloud"                    # OpenAI/Anthropic/etc (futuro)


class ErrorKind(Enum):
    """Clasificación de errores LLM. Determina si rotamos o subimos a la app."""

    RATE_LIMIT = "rate_limit"      # 429, "throttled", "concurrency limit"
    TIMEOUT = "timeout"            # connect timeout, read timeout
    IDLE_TIMEOUT = "idle_timeout"  # stream sin chunks por N segundos
    BILLING = "billing"            # 402, "insufficient credits"
    FORMAT = "format"              # JSON inválido, schema mismatch
    AUTH = "auth"                  # 401, 403 (no rotamos: requiere acción humana)
    PERMANENT = "permanent"        # cualquier otro error definitivo

    def is_failover_worthy(self) -> bool:
        """¿Este error justifica rotar al siguiente candidato?"""
        return self in {
            ErrorKind.RATE_LIMIT,
            ErrorKind.TIMEOUT,
            ErrorKind.IDLE_TIMEOUT,
            ErrorKind.BILLING,
            ErrorKind.FORMAT,
        }


@dataclass
class LLMEndpoint:
    """Un endpoint LLM concreto en la candidate chain."""

    id: str
    kind: EndpointKind
    client: Any  # FastRouter | HttpReasoner | LLMReasoner | otro
    priority: int  # menor = primero
    timeout_s: float = 30.0
    idle_timeout_s: Optional[float] = None  # None = sin watchdog
    max_tokens_default: int = 256


@dataclass
class CooldownState:
    """Estado de cooldown persistido por endpoint."""

    endpoint_id: str
    error_count: int = 0
    cooldown_until: float = 0.0  # epoch seconds; 0 = no cooldown
    last_used: float = 0.0
    last_error_kind: Optional[ErrorKind] = None

    def to_dict(self) -> dict:
        return {
            "endpoint_id": self.endpoint_id,
            "error_count": self.error_count,
            "cooldown_until": self.cooldown_until,
            "last_used": self.last_used,
            "last_error_kind": self.last_error_kind.value if self.last_error_kind else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CooldownState:
        kind_raw = d.get("last_error_kind")
        return cls(
            endpoint_id=d["endpoint_id"],
            error_count=d.get("error_count", 0),
            cooldown_until=d.get("cooldown_until", 0.0),
            last_used=d.get("last_used", 0.0),
            last_error_kind=ErrorKind(kind_raw) if kind_raw else None,
        )


@dataclass
class RouterResult:
    """Resultado de una invocación exitosa del router."""

    text: str
    endpoint_id: str
    attempts: int
    elapsed_ms: float
    metadata: dict = field(default_factory=dict)
