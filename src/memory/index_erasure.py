"""Cleanup-only policy-index access, including while semantic recall is disabled."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from contextvars import copy_context
from functools import partial
from pathlib import Path
import threading

from src.core.owned_work import OwnedBlockingWork
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureParticipant, ErasureResult, ErasureStep
from src.memory.index_deletions import IndexDeletionBatch
from src.memory.index_maintenance import IndexMaintenance
from src.memory.privacy_services import PrivacyServices


class PolicyIndexEraser:
    """Sweep erased policy IDs without embeddings, new collections or legacy edits."""

    def __init__(self, privacy: PrivacyServices, descriptor: dict, *, memory=None, client_factory=None):
        self.privacy = privacy
        if memory is not None and memory._privacy is not privacy:
            raise ValueError('index_cleanup_authority_mismatch')
        self.descriptor = memory.index_descriptor if memory is not None and memory.index_descriptor is not None else descriptor
        self._ledger = privacy.ledger
        # Use the writer's barrier: an upsert read before revocation must finish
        # before cleanup acknowledges removal of that same physical index.
        self._index_lock = memory._index_lock if memory is not None else threading.Lock()
        self._locks: dict[tuple, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._client_factory = client_factory
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="index-erasure")
        self._work = OwnedBlockingWork(worker=self._offload, capacity=8)
        self._closing: asyncio.Task | None = None
        self.maintenance = IndexMaintenance(privacy, self.clean_deletions)

    async def _offload(self, function, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(self._executor,
            copy_context().run, partial(function, *args, **kwargs))

    def _lock_for(self, descriptor: dict):
        if descriptor == self.descriptor:
            return self._index_lock
        key = (descriptor.get('path'), descriptor.get('collection'))
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _open(self, descriptor: dict):
        path, collection = descriptor.get("path"), descriptor.get("collection")
        if set(descriptor) != {"path", "collection"} or type(path) is not str or not Path(path).is_absolute() or type(collection) is not str or not collection.endswith("_policy_v1"):
            raise ValueError("invalid_policy_index_target")
        target = Path(path)
        # Missing files need no client, directory creation, model import or I/O
        # beyond this worker's existence check. Permission errors must fail.
        try:
            target.stat()
        except FileNotFoundError:
            return None, None
        if not target.is_dir():
            raise ValueError("invalid_policy_index_directory")
        import chromadb
        from chromadb.config import Settings
        from chromadb.errors import NotFoundError
        factory = self._client_factory or chromadb.PersistentClient
        client = factory(path=path, settings=Settings(anonymized_telemetry=False))
        try:
            return client, client.get_collection(name=collection, embedding_function=None)
        except NotFoundError:
            client.close()
            return None, None
        except BaseException:
            client.close()
            raise

    def _clean(self, step: ErasureStep) -> ErasureResult:
        if step.participant is not ErasureParticipant.SEMANTIC_INDEX or step.version != 1:
            raise ValueError("unsupported_policy_index_target")
        with self._lock_for(step.descriptor):
            rows = self._ledger.erasures.deleted_facts(step.cursor)
            pending = self._ledger.index_deletions.unfinished(step.target_id, rows)
            if pending:
                self._delete_rows(step.descriptor, [row['id'] for row in pending])
                self._ledger.index_deletions.record_completed(step.target_id, pending)
            outstanding = self._ledger.index_deletions.pending_count(step.target_id)
            # The per-target journal includes quarantine and any unknown SOT
            # state that a deleted-row cursor alone cannot prove cleaned.
            cursor = rows[-1]['id'] if len(rows)==32 else None
            return ErasureResult(outstanding==0, cursor, outstanding)

    def _delete_rows(self, descriptor: dict, ids: list[str]) -> None:
        client, collection = self._open(descriptor)
        try:
            if collection is not None:
                collection.delete(ids=ids)
        finally:
            if client is not None:
                client.close()

    def _clean_deletions(self, batch: IndexDeletionBatch) -> None:
        with self._lock_for(batch.descriptor):
            items = self._ledger.index_deletions.eligible(batch)
            if batch.version != 1:
                raise ValueError('unsupported_policy_index_target')
            if items:
                self._delete_rows(batch.descriptor, [item.fact_id for item in items])
            try:
                # Publish physical proof before releasing the writer barrier so
                # conversation cleanup can reuse it without a duplicate delete.
                self._ledger.index_deletions.finish(batch)
            except Exception:
                # Physical work finished. The retained scheduler outcome retries
                # this receipt rather than running native cleanup again.
                return

    async def clean_deletions(self, batch: IndexDeletionBatch) -> None:
        """Run cleanup-only batches under the same physical writer barrier."""
        await self._work.run(self._clean_deletions, batch)

    async def clean(self, step: ErasureStep) -> ErasureResult:
        """Retain actual worker ownership even if a cleanup observer disappears."""
        return await self._work.run(self._clean, step)

    def begin_close(self) -> asyncio.Task:
        """Drain native index work before releasing its dedicated executor."""
        if self._closing is None:
            maintenance = self.maintenance.begin_close()
            async def finish():
                await maintenance
                await self._work.begin_close()
                await asyncio.to_thread(self._executor.shutdown, wait=True)
            self._closing = asyncio.create_task(finish())
        return self._closing


async def register_policy_index_target(config: dict, privacy: PrivacyServices) -> dict:
    """Persist the configured location before any optional semantic writer opens it."""
    settings = config.get("memory", {})
    descriptor = await privacy.acall(lambda: {
        "path": str(Path(settings.get("chroma_path", "./data/memory_db")).resolve()),
        "collection": settings.get("long_term_collection", "user_memories") + "_policy_v1"})
    await privacy.acall(privacy.ledger.erasures.register_target, ErasureParticipant.SEMANTIC_INDEX, 1, descriptor)
    return descriptor


async def build_erasure_services(config: dict, *, privacy: PrivacyServices, shutdown: AsyncExitStack,
                                 memory=None, client_factory=None) -> ErasureCoordinator:
    """Register durable index targets without enabling recall or broad erase UI."""
    owner = ErasureCoordinator(privacy)
    shutdown.push_async_callback(owner.aclose)
    descriptor = await register_policy_index_target(config, privacy)
    driver = PolicyIndexEraser(privacy, descriptor, memory=memory, client_factory=client_factory)
    owner.register(ErasureParticipant.SEMANTIC_INDEX, driver.clean,
                   versions=frozenset({1}), begin_close=driver.begin_close)
    driver.maintenance.start()
    owner.start()
    return owner
