"""
LLM Module
Modelos de lenguaje para razonamiento y clasificacion.
"""

from src.llm.adapters import FastRouterAdapter, HttpReasonerAdapter
from src.llm.buffered_streamer import (
    BufferedLLMStreamer,
    BufferConfig,
    BufferStrategy,
    ConversationStreamer,
    create_buffered_streamer,
)
from src.llm.cooldown import CooldownManager
from src.llm.error_classifier import IdleTimeoutError, classify_error
from src.llm.idle_watchdog import idle_watchdog
from src.llm.reasoner import LLMReasoner, FastRouter, HttpReasoner
from src.llm.router import FailedAttempt, FallbackSummaryError, LLMRouter
from src.llm.router_factory import build_llm_router_from_config
from src.llm.types import (
    CooldownState,
    EndpointKind,
    ErrorKind,
    LLMEndpoint,
    RouterResult,
)

__all__ = [
    # Reasoners (legacy)
    "LLMReasoner",
    "HttpReasoner",
    "FastRouter",
    # Buffered streaming (legacy)
    "BufferedLLMStreamer",
    "BufferConfig",
    "BufferStrategy",
    "ConversationStreamer",
    "create_buffered_streamer",
    # Router + types
    "LLMRouter",
    "LLMEndpoint",
    "EndpointKind",
    "ErrorKind",
    "CooldownState",
    "RouterResult",
    "FallbackSummaryError",
    "FailedAttempt",
    # Manager
    "CooldownManager",
    # Adapters
    "FastRouterAdapter",
    "HttpReasonerAdapter",
    # Helpers
    "build_llm_router_from_config",
    "idle_watchdog",
    "IdleTimeoutError",
    "classify_error",
]
