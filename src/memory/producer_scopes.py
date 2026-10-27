"""Metadata-only affinity of controllers and their actual retained child work."""

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4
from weakref import WeakMethod

from src.memory.fact_ledger import FactLedger, MemorySource


def _validate_sources(sources):
    if sources is not None and (type(sources) is not tuple or len(sources) > 256
                                or not all(isinstance(source, MemorySource) for source in sources)):
        raise ValueError('invalid_producer_scope_sources')


@dataclass(eq=False)
class ProducerScope:
    """No raw values, owner objects, task results or exception references."""

    token: str = field(default_factory=lambda: str(uuid4()))
    sources: tuple[MemorySource, ...] | None = None
    active: bool = True

    def bind(self, sources: tuple[MemorySource, ...] | None) -> None:
        """Publish metadata to an exclusive receiver before returning raw data.

        The caller must retire any earlier raw locals before refining this
        receipt. Binding cannot extend ownership, retention or authority.
        """
        _validate_sources(sources)
        self.sources = sources


_CURRENT_SCOPE = ContextVar('owned_producer_scope', default=None)


class ProducerScopes:
    """Keep source affinity after a cancelled parent releases its own raw locals.

    Actual child tasks inherit only the receipt. Their existing native owners
    retain the work; this registry holds metadata and a weak snapshot callback.
    It is used on the event loop, never inside an input device callback.
    """

    def __init__(self, operations: Callable[[], tuple[asyncio.Task, ...]], *, capacity: int = 8):
        if type(capacity) is not int or not 1 <= capacity <= 256:
            raise ValueError('invalid_producer_scope_capacity')
        self._operations = WeakMethod(operations)
        self._capacity = capacity
        self._scopes: dict[str, ProducerScope] = {}

    def _reap(self):
        snapshot = self._operations()
        if snapshot is None:
            # Losing the owner is not proof that its native work has exited.
            return
        # Include done tasks until the actual owner's retirement callback
        # releases them: their result may still carry raw data at that handoff.
        retained = {scope.token for task in snapshot()
                    if (scope := task.get_context().get(_CURRENT_SCOPE)) is not None}
        for token, scope in tuple(self._scopes.items()):
            if not scope.active and token not in retained:
                del self._scopes[token]

    @staticmethod
    def _validate(sources):
        _validate_sources(sources)

    @contextmanager
    def operation(self, sources: tuple[MemorySource, ...] | None = None) -> Iterator[ProducerScope]:
        """Reserve affinity before raw acquisition; None means not known yet."""
        self._validate(sources)
        self._reap()
        if len(self._scopes) >= self._capacity:
            raise RuntimeError('producer_scope_capacity')
        scope = ProducerScope(sources=sources)
        self._scopes[scope.token] = scope
        token = _CURRENT_SCOPE.set(scope)
        try:
            yield scope
        finally:
            scope.active = False
            _CURRENT_SCOPE.reset(token)
            self._reap()

    def replace_sources(self, sources: tuple[MemorySource, ...] | None) -> None:
        """Resolve/refine affinity only after excluded raw locals are retired."""
        self._validate(sources)
        current = _CURRENT_SCOPE.get()
        if current is not None and self._scopes.get(current.token) is current:
            current.sources = sources

    def busy(self, conversation_id: str, ledger: FactLedger) -> bool:
        """Unknown reads stay pending; known unrelated work does not block."""
        if not ledger.authority_available:
            raise RuntimeError('privacy_authority_unavailable')
        self._reap()
        return any(scope.sources is None or any(source.conversation_id == conversation_id
                   or ledger.dependencies_committed_unavailable(source.fact_dependencies)
                   for source in scope.sources) for scope in self._scopes.values())

    def resource_status(self) -> dict:
        """Expose bounded counters without lineage, raw values or task objects."""
        self._reap()
        return {'scopes': len(self._scopes), 'capacity': self._capacity,
                'unknown': sum(scope.sources is None for scope in self._scopes.values())}
