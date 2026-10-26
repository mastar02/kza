"""Immutable prompt text with the actual raw and semantic inputs it depends on."""

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, replace
from typing import Any

from src.memory.fact_ledger import FactDependency, MemorySource


def merge_sources(*groups: tuple[MemorySource, ...]) -> tuple[MemorySource, ...]:
    """Combine matching captures without dropping dependencies or renewing expiry."""
    merged = {}
    for group in groups:
        for source in group:
            previous = merged.get(source.capture_id)
            if previous is None:
                merged[source.capture_id] = source
                continue
            if (previous.conversation_id, previous.captured_at, previous.origin) != (source.conversation_id, source.captured_at, source.origin):
                raise ValueError("conflicting_capture_provenance")
            dependencies = _merge_facts(previous.fact_dependencies, source.fact_dependencies)
            merged[source.capture_id] = replace(previous, expires_at=min(previous.expires_at, source.expires_at),
                                                 fact_dependencies=dependencies)
    return tuple(merged.values())


def _merge_facts(*groups: tuple[FactDependency, ...]) -> tuple[FactDependency, ...]:
    merged = {}
    for group in groups:
        for dependency in group:
            previous = merged.get(dependency.fact_id)
            if previous is not None and previous != dependency:
                raise ValueError("conflicting_fact_dependency")
            merged[dependency.fact_id] = dependency
    return tuple(merged.values())


@dataclass(frozen=True)
class PromptContext:
    """Keep provenance beside the exact selected text through async prompt building."""

    text: str
    sources: tuple[MemorySource, ...] = ()
    facts: tuple[FactDependency, ...] = ()

    def bind_sources(self, captures: tuple[MemorySource, ...]) -> tuple[MemorySource, ...]:
        """Bind semantic receipts to current captures and retain included raw inputs."""
        if self.facts and not captures:
            raise PermissionError("prompt_capture_provenance_required")
        bound = []
        for source in captures:
            dependencies = _merge_facts(source.fact_dependencies, self.facts)
            expiry = min([source.expires_at, *(d.expires_at for d in dependencies if d.expires_at is not None)])
            bound.append(replace(source, expires_at=expiry, fact_dependencies=dependencies))
        return merge_sources(tuple(bound), self.sources)


class SourcesInvalidated(RuntimeError):
    """An async inference or delivery lost one of its admitted prompt inputs."""


async def run_private(operation: Coroutine, available: Callable[[], bool]) -> Any:
    """Cancel invalidated work and retain its cleanup through caller cancellation."""
    if not available():
        operation.close()
        raise SourcesInvalidated("prompt_sources_invalidated")

    async def watch():
        while available():
            await asyncio.sleep(.01)

    task = asyncio.create_task(operation)
    monitor = asyncio.create_task(watch())
    try:
        await asyncio.wait({task, monitor}, return_when=asyncio.FIRST_COMPLETED)
        if not available():
            raise SourcesInvalidated("prompt_sources_invalidated")
        result = await task
    finally:
        for child in (task, monitor):
            if not child.done():
                child.cancel()

        async def settle():
            await asyncio.gather(task, monitor, return_exceptions=True)
        closing = asyncio.create_task(settle())
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
        closing.result()
        if cancelled:
            raise asyncio.CancelledError
    if not available():
        raise SourcesInvalidated("prompt_sources_invalidated")
    return result
