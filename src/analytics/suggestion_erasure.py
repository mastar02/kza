"""Durable proposal cleanup and actual pre-marker producer/voice read barriers."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from functools import partial
from weakref import WeakSet, ref

from src.analytics.suggestion_engine import SuggestionEngine
from src.analytics.suggestion_storage import (
    RUNTIME_TARGET,
    cleanup_suggestions,
    suggestion_descriptor,
)
from src.core.owned_work import OwnedBlockingWork
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureParticipant, ErasureResult, ErasureStep
from src.memory.privacy_services import PrivacyServices


@dataclass
class RuntimeBarrier:
    """Only live task ownership and content-free delivery lifetime tokens."""

    tasks: set[asyncio.Task] = field(default_factory=set)
    deliveries: list[tuple[ref, frozenset[int]]] = field(default_factory=list)

    def track(self, tasks) -> None:
        for task in tasks:
            if not task.done() and task not in self.tasks:
                self.tasks.add(task)
                # Never retain a finished native result/traceback until a retry.
                task.add_done_callback(self.tasks.discard)

    def pending(self) -> int:
        remaining = []
        for owner, tokens in self.deliveries:
            handler = owner()
            if handler is not None:
                live = tokens & handler.retained_delivery_scopes()
                if live:
                    remaining.append((owner, live))
        self.deliveries = remaining
        return sum(not task.done() for task in self.tasks) + sum(
            len(tokens) for _, tokens in self.deliveries
        )


class SuggestionEraser:
    """Clean registered sidecars even with analytics off; wait for real producers."""

    def __init__(self, privacy: PrivacyServices):
        self.privacy = privacy
        self._engines, self._routers = WeakSet(), WeakSet()
        self._barriers: dict[str, RuntimeBarrier] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="suggestion-erasure"
        )
        self._disk = OwnedBlockingWork(worker=self._offload)
        self._closing = None

    async def _offload(self, function, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, copy_context().run, partial(function, *args, **kwargs)
        )

    def attach_engine(self, engine: SuggestionEngine) -> None:
        """Bind the actual native owner to the leased privacy authority."""
        if (
            not isinstance(engine, SuggestionEngine)
            or engine.store.policy is not self.privacy.ledger
        ):
            raise ValueError("suggestion_engine_authority_mismatch")
        if engine not in self._engines and engine.event_logger.snapshot_operations():
            raise ValueError("suggestion_owner_requires_registration_before_work")
        self._engines.add(engine)
        self.privacy.add_borrower(engine.event_logger.begin_close)

    def attach_router(self, router) -> None:
        """Bind scoped voice flows, excluding unsupported backend ownership."""
        if router._source_policy is not self.privacy.ledger:
            raise ValueError("suggestion_router_authority_mismatch")
        if router not in self._routers and (
            router.snapshot_suggestion_flows() or router._pending_suggestion is not None
        ):
            raise ValueError("suggestion_router_requires_registration_before_work")
        if router.suggestion_engine is not None:
            self.attach_engine(router.suggestion_engine)
            from src.pipeline.response_handler import ResponseHandler

            if router.response_handler is not None and not isinstance(
                router.response_handler, ResponseHandler
            ):
                raise ValueError("unsupported_suggestion_response_owner")
        self._routers.add(router)
        self.privacy.add_borrower(router.begin_close_suggestions)

    def _snapshot_runtime(self) -> RuntimeBarrier:
        barrier = RuntimeBarrier()
        for engine in self._engines:
            barrier.track(engine.event_logger.snapshot_operations())
        for router in self._routers:
            flows = router.snapshot_suggestion_flows()
            barrier.track(flows)
            if (
                router.suggestion_engine is not None
                and router.response_handler is not None
            ):
                barrier.deliveries.append(
                    (
                        ref(router.response_handler),
                        router.response_handler.retained_delivery_scopes(),
                    )
                )
            for task in flows:
                if not task.done() and not task.cancelling():
                    task.cancel()
            router.prune_private()
        return barrier

    async def clean(self, step: ErasureStep) -> ErasureResult:
        """A pending cohort does not block cleanup of other registered targets."""
        if step.participant is not ErasureParticipant.SUGGESTIONS or step.version != 1:
            raise ValueError("unsupported_suggestion_target")
        targets = await self.privacy.acall(
            self.privacy.ledger.erasures.targets, ErasureParticipant.SUGGESTIONS
        )
        if not any(
            target["id"] == step.target_id
            and target["version"] == step.version
            and target["descriptor"] == step.descriptor
            for target in targets
        ):
            raise ValueError("unregistered_suggestion_target")
        ledger = self.privacy.ledger
        if not ledger.authority_available or not ledger.conversation_committed_revoked(
            step.conversation_id
        ):
            raise RuntimeError("suggestion_erasure_authority_unavailable")
        if step.descriptor.get("kind") == "runtime":
            if step.descriptor != RUNTIME_TARGET or step.cursor is not None:
                raise ValueError("unsupported_suggestion_runtime_target")
            barrier = self._barriers.get(step.operation_id)
            if barrier is None:
                barrier = self._barriers[step.operation_id] = self._snapshot_runtime()
            for router in self._routers:
                router.prune_private()
            pending = barrier.pending()
            if not pending:
                self._barriers.pop(step.operation_id, None)
            return ErasureResult(pending == 0, pending_count=pending)
        return await self._disk.run(
            cleanup_suggestions,
            ledger,
            step.descriptor,
            step.conversation_id,
            step.cursor,
        )

    def begin_close(self) -> asyncio.Task:
        """Retain native callbacks, scoped voice cleanup and their actual workers."""
        if self._closing is None:
            barrier = self._snapshot_runtime()
            drains = [router.begin_close_suggestions() for router in self._routers]
            disk = self._disk.begin_close()

            async def finish():
                await asyncio.gather(disk, *drains)
                while barrier.pending() or any(
                    b.pending() for b in self._barriers.values()
                ):
                    await asyncio.sleep(0.005)
                self._barriers.clear()
                await asyncio.to_thread(self._executor.shutdown, wait=True)

            self._closing = asyncio.create_task(finish())
        return self._closing


async def build_suggestion_erasure(
    config: dict, *, privacy: PrivacyServices, owner: ErasureCoordinator
) -> SuggestionEraser:
    """Register physical/runtime destinations before optional producer admission."""
    descriptor = await privacy.acall(
        suggestion_descriptor,
        config.get("analytics", {}).get("suggestions_path", "./data/suggestions.json"),
    )
    for target in (descriptor, RUNTIME_TARGET):
        await privacy.acall(
            privacy.ledger.erasures.register_target,
            ErasureParticipant.SUGGESTIONS,
            1,
            target,
        )
    driver = SuggestionEraser(privacy)
    owner.register(
        ErasureParticipant.SUGGESTIONS,
        driver.clean,
        versions=frozenset({1}),
        begin_close=driver.begin_close,
    )
    return driver
