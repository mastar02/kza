"""Registered snapshot cleanup and retained runtime conversation context scopes."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from weakref import WeakSet

from src.core.owned_work import OwnedBlockingWork
from src.memory.erasure_journal import ErasureParticipant, ErasureResult, ErasureStep
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.privacy_services import PrivacyServices
from src.orchestrator.context_storage import cleanup_snapshots, register_snapshots
from src.orchestrator.context_retention import ContextRetention


async def _await(function, *args, **kwargs):
    return await function(*args, **kwargs)


class ContextEraser:
    """Connect known context buffers and policy roots without broad erasure claims."""

    def __init__(self, privacy: PrivacyServices):
        self.privacy = privacy
        self._managers = WeakSet()
        self._short_terms = WeakSet()
        self._routers = WeakSet()
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="context-erasure"
        )
        self._disk = OwnedBlockingWork(worker=self._offload)
        self._runtime = OwnedBlockingWork(worker=_await)
        self._closing = None
        self.retention = ContextRetention(privacy, runtime_sweep=self._sweep_runtime)

    async def _offload(self, function, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, copy_context().run, partial(function, *args, **kwargs)
        )

    def attach_manager(self, manager) -> None:
        """Attach before producer admission and retain its actual native shutdown."""
        if manager.source_policy is not self.privacy.ledger:
            raise ValueError("context_authority_mismatch")
        if (
            manager.persister is not None
            and manager.persister.source_policy is not self.privacy.ledger
        ):
            raise ValueError("context_persister_authority_mismatch")
        self._managers.add(manager)
        manager.retention_notify = self.retention.notify
        if manager.persister is not None:
            manager.persister.retention_notify = self.retention.notify
        self.retention.notify()
        self.privacy.add_borrower(manager.begin_close)

    def attach_short_term(self, memory) -> None:
        """Bind the non-orchestrated history to the same source authority."""
        if memory._source_policy is not self.privacy.ledger:
            raise ValueError("short_term_authority_mismatch")
        self._short_terms.add(memory)
        memory.retention_notify = self.retention.notify
        self.retention.notify()

    def attach_router(self, router) -> None:
        """Include idle query/classifier/queue copies; active producers stay separate."""
        if router._source_policy is not self.privacy.ledger:
            raise ValueError("context_router_authority_mismatch")
        self._routers.add(router)
        self.retention.notify()

    async def _sweep_runtime(self) -> dict:
        now = self.privacy.ledger.clock()
        results = []
        for manager in list(self._managers):
            results.append(await manager.prune_retained())
        for memory in list(self._short_terms):
            results.append(memory.prune_retained())
        for router in list(self._routers):
            router.prune_private()
        return {'checked_at':now,'next_at':min([now+30,*(r['next_at'] for r in results)]),
            'pruned':sum(r['pruned'] for r in results),'unknown':sum(r['unknown'] for r in results),
            'pending_operations':sum(r['pending_operations'] for r in results)}

    async def _clean_runtime(self, step: ErasureStep) -> ErasureResult:
        if step.descriptor != {"kind": "runtime", "name": "conversation_buffers"}:
            raise ValueError("unsupported_context_runtime_target")
        if not self.privacy.ledger.conversation_revoked(step.conversation_id):
            raise RuntimeError("context_erasure_authority_unavailable")
        unknown = 0
        for manager in list(self._managers):
            unknown += await manager.erase_conversation(step.conversation_id)
        for memory in list(self._short_terms):
            unknown += memory.erase_conversation(step.conversation_id)
        for router in list(self._routers):
            router.prune_private()
        return ErasureResult(unknown == 0, pending_count=unknown)

    async def clean(self, step: ErasureStep) -> ErasureResult:
        """Keep native file work and compaction cancellation owned after timeout."""
        if step.participant is not ErasureParticipant.CONTEXTS or step.version != 1:
            raise ValueError("unsupported_context_target")
        targets = await self.privacy.acall(
            self.privacy.ledger.erasures.targets, ErasureParticipant.CONTEXTS
        )
        if not any(
            row["id"] == step.target_id
            and row["version"] == step.version
            and row["descriptor"] == step.descriptor
            for row in targets
        ):
            raise ValueError("unregistered_context_target")
        if step.descriptor.get("kind") == "runtime":
            return await self._runtime.run(self._clean_runtime, step)
        return await self._disk.run(
            cleanup_snapshots,
            self.privacy.ledger,
            step.descriptor,
            step.conversation_id,
            step.cursor,
        )

    def begin_close(self) -> asyncio.Task:
        """Drain actual callbacks before releasing their private native pool."""
        if self._closing is None:
            retention = self.retention.begin_close()
            disk, runtime = self._disk.begin_close(), self._runtime.begin_close()

            async def finish():
                await retention
                await asyncio.gather(disk, runtime)
                await asyncio.to_thread(self._executor.shutdown, wait=True)

            self._closing = asyncio.create_task(finish())
        return self._closing


async def build_context_erasure(
    config: dict, *, privacy: PrivacyServices, owner: ErasureCoordinator, memory=None
) -> ContextEraser:
    """Register historical roots even with persistence, recall or orchestration off."""
    base = (
        config.get("orchestrator", {})
        .get("context", {})
        .get("persistence", {})
        .get("base_path", "data/contexts")
    )
    await privacy.acall(register_snapshots, privacy.ledger, base)
    await privacy.acall(
        privacy.ledger.erasures.register_target,
        ErasureParticipant.CONTEXTS,
        1,
        {"kind": "runtime", "name": "conversation_buffers"},
    )
    driver = ContextEraser(privacy)
    owner.register(
        ErasureParticipant.CONTEXTS,
        driver.clean,
        versions=frozenset({1}),
        begin_close=driver.begin_close,
    )
    if memory is not None:
        driver.attach_short_term(memory.short_term)
    await driver.retention.refresh()
    driver.retention.start()
    return driver
