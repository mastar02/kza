"""
Orchestrator Module
Sistema de orquestacion para manejar multiples usuarios concurrentemente.

Componentes:
- ContextManager: Mantiene contexto de conversacion por usuario
- PriorityQueue: Cola priorizada para peticiones al LLM
- RequestDispatcher: Enruta peticiones al path correcto (fast/slow)
- CancellationToken: Permite cancelar peticiones en curso

Arquitectura:
                    ┌─────────────────────────────┐
                    │     REQUEST DISPATCHER      │
                    └─────────────┬───────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
          ┌───────────────┐            ┌───────────────────┐
          │   FAST PATH   │            │    SLOW PATH      │
          │  (paralelo)   │            │  (serializado)    │
          │               │            │                   │
          │ - Domotica    │            │ ┌───────────────┐ │
          │ - Router 7B   │            │ │ContextManager │ │
          │ - Rutinas     │            │ └───────┬───────┘ │
          └───────────────┘            │         │         │
                                       │ ┌───────▼───────┐ │
                                       │ │PriorityQueue  │ │
                                       │ └───────┬───────┘ │
                                       │         │         │
                                       │ ┌───────▼───────┐ │
                                       │ │   LLM 32B     │ │
                                       │ └───────────────┘ │
                                       └───────────────────┘
"""

from src.orchestrator.context_manager import (
    UserContext,
    ContextManager,
    ConversationTurn,
    MusicPreferences
)
from src.orchestrator.context_persistence import PersistentContextManager
from src.orchestrator.priority_queue import (
    Priority,
    Request,
    RequestStatus,
    PriorityRequestQueue
)
from src.orchestrator.cancellation import (
    CancellationToken,
    CancellationScope
)
from src.orchestrator.dispatcher import (
    RequestDispatcher,
    MultiUserOrchestrator,
    DispatchResult,
    PathType
)
from src.orchestrator.compactor import (
    Compactor,
    CompactionResult,
    CompactionError,
)
from src.orchestrator.context_persister import ContextPersister, PERSISTED_VERSION

__all__ = [
    # Context
    "UserContext",
    "ContextManager",
    "PersistentContextManager",
    "ConversationTurn",
    "MusicPreferences",
    # Queue
    "Priority",
    "Request",
    "RequestStatus",
    "PriorityRequestQueue",
    # Cancellation
    "CancellationToken",
    "CancellationScope",
    # Dispatcher
    "RequestDispatcher",
    "MultiUserOrchestrator",
    "DispatchResult",
    "PathType",
    # Plan #2 OpenClaw
    "Compactor",
    "CompactionResult",
    "CompactionError",
    "ContextPersister",
    "PERSISTED_VERSION",
]
