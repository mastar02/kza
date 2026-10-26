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
from src.memory.privacy_services import PrivacyServices


class PolicyIndexEraser:
    """Sweep erased policy IDs without embeddings, new collections or legacy edits."""

    def __init__(self, privacy: PrivacyServices, descriptor: dict, *, memory=None, client_factory=None):
        self.privacy, self.descriptor = privacy, descriptor
        self._ledger = privacy.ledger
        # Use the writer's barrier: an upsert read before revocation must finish
        # before cleanup acknowledges removal of that same physical index.
        self._index_lock = memory._index_lock if memory is not None else threading.Lock()
        self._client_factory = client_factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="index-erasure")
        self._work = OwnedBlockingWork(worker=self._offload, capacity=4)
        self._closing: asyncio.Task | None = None

    async def _offload(self, function, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(self._executor,
            copy_context().run, partial(function, *args, **kwargs))

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
        with self._index_lock:
            rows = self._ledger.erasures.deleted_facts(step.cursor)
            client, collection = self._open(step.descriptor)
            try:
                if collection is not None and rows:
                    collection.delete(ids=[row["id"] for row in rows])
            finally:
                if client is not None:
                    client.close()
            if step.descriptor == self.descriptor:
                for row in rows:
                    self._ledger.acknowledge(row["id"], row["revision"])
            return ErasureResult(len(rows) < 32, rows[-1]["id"] if rows else step.cursor)

    async def clean(self, step: ErasureStep) -> ErasureResult:
        """Retain actual worker ownership even if a cleanup observer disappears."""
        return await self._work.run(self._clean, step)

    def begin_close(self) -> asyncio.Task:
        """Drain native index work before releasing its dedicated executor."""
        if self._closing is None:
            drain = self._work.begin_close()
            async def finish():
                await drain
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
    owner.start()
    return owner
