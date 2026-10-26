"""Content-free lineage inherited by actual producer tasks and native workers."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from src.memory.fact_ledger import MemorySource


@dataclass(frozen=True)
class ProducerContext:
    """Identify an admitted capture scope without retaining text or audio."""

    room_id: str | None = None
    sources: tuple[MemorySource, ...] = ()


PRODUCER_CONTEXT = ContextVar("producer_context", default=ProducerContext())


@contextmanager
def producer_context(room_id: str | None, sources: tuple[MemorySource, ...] = ()):
    """Bind metadata while creating the actual tasks that own private input."""
    token = PRODUCER_CONTEXT.set(ProducerContext(room_id, sources))
    try:
        yield
    finally:
        PRODUCER_CONTEXT.reset(token)
