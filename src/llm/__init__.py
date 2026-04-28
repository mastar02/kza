"""
LLM Module
Modelos de lenguaje para razonamiento y clasificacion.
"""

from src.llm.reasoner import LLMReasoner, FastRouter, HttpReasoner
from src.llm.buffered_streamer import (
    BufferedLLMStreamer,
    BufferConfig,
    BufferStrategy,
    ConversationStreamer,
    create_buffered_streamer,
)
from src.llm.types import (
    EndpointKind,
    ErrorKind,
    LLMEndpoint,
    CooldownState,
    RouterResult,
)
from src.llm.cooldown import CooldownManager
from src.llm.router import LLMRouter, FallbackSummaryError
from src.llm.idle_watchdog import idle_watchdog, IdleTimeoutError
from src.llm.router_factory import build_llm_router

__all__ = [
    "LLMReasoner",
    "HttpReasoner",
    "FastRouter",
    "BufferedLLMStreamer",
    "BufferConfig",
    "BufferStrategy",
    "ConversationStreamer",
    "create_buffered_streamer",
    # Failover / Routing (plan #1 OpenClaw)
    "EndpointKind",
    "ErrorKind",
    "LLMEndpoint",
    "CooldownState",
    "RouterResult",
    "CooldownManager",
    "LLMRouter",
    "FallbackSummaryError",
    "idle_watchdog",
    "IdleTimeoutError",
    "build_llm_router",
]
