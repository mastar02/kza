"""Policy-owned long-term memory, with a rebuildable local vector index."""

import asyncio
import fcntl
import logging
import math
import os
import threading
from pathlib import Path

from src.memory.fact_ledger import FactLedger, MemoryReviewer, MemorySource

logger = logging.getLogger(__name__)


class MemoryCloseTimeout(RuntimeError):
    """A native memory operation still owns resources after close deadline."""


class LongTermMemory:
    """Use ledger state for every read; keep legacy facts out of inference.

    Async application consumers use the ``a*`` methods. Synchronous methods
    remain worker adapters to the same policy, including older integrations.
    No method imports or rewrites the pre-policy collection automatically.
    """

    def __init__(self, chroma_client, collection_name: str = "user_memories", *,
                 ledger_path: str = ":memory:", clock=None, automatic_days: float = 30,
                 shutdown_timeout_s: float = 1):
        if type(shutdown_timeout_s) not in (int, float) or not math.isfinite(shutdown_timeout_s) or shutdown_timeout_s <= 0:
            raise ValueError("invalid_memory_shutdown_timeout")
        self._client, self._collection_name = chroma_client, collection_name
        self._collection = None
        self.index_collection_name = collection_name + "_policy_v1"
        self.ledger = FactLedger(ledger_path, clock=clock, automatic_days=automatic_days)
        self.shutdown_timeout_s = shutdown_timeout_s
        self._initializing = threading.RLock()
        self._index_lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._resources_lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(8)
        self._operations: set[asyncio.Task] = set()
        self._closed = False
        self._lease: int | None = None
        self._close_task: asyncio.Task | None = None
        self._maintenance_event: asyncio.Event | None = None
        self._maintenance_loop: asyncio.AbstractEventLoop | None = None

    def _notify(self):
        if self._maintenance_loop is not None:
            self._maintenance_loop.call_soon_threadsafe(self._maintenance_event.set)

    def _require_open(self):
        if self._closed:
            raise RuntimeError("memory_closed")

    def initialize(self) -> None:
        """Create only the new policy index and acquire one writer lease."""
        with self._initializing:
            self._require_open()
            if self._collection is not None:
                return
            if self.ledger.path != ":memory:" and self._lease is None:
                Path(self.ledger.path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                descriptor = os.open(str(Path(self.ledger.path)) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    os.close(descriptor)
                    raise RuntimeError("memory_writer_already_active") from None
                self._lease = descriptor
            # Acquire the writer lease before schema migration can mutate data.
            self.ledger.initialize()
            self._collection = self._client.get_or_create_collection(
                name=self.index_collection_name,
                metadata={"description": "Policy-managed memory index", "fact_policy": 1},
            )

    def store_fact(self, fact: str, category: str, confidence: float = .8,
                   metadata: dict | None = None, *, sources: tuple[MemorySource, ...] = ()) -> str:
        """Persist a sourced 30-day automatic fact; metadata cannot confirm it."""
        self._require_open()
        if metadata and set(metadata) - {"origin"}:
            raise ValueError("memory_metadata_is_server_owned")
        self.initialize()
        stored = self.ledger.store(fact, category, confidence, sources)
        self._notify()
        self.reconcile()
        return stored.id

    def propose(self, content: str, sources: tuple[MemorySource, ...], *, category="fact") -> dict:
        """Propose permanent memory, retaining it only until authenticated review."""
        self.initialize()
        proposal = self.ledger.store(content, category, 1.0, sources, proposed=True)
        self._notify()
        return proposal.to_dict()

    def confirm(self, fact_id: str, actor: MemoryReviewer) -> dict:
        """Apply an authenticated review and queue its durable index update."""
        self.initialize()
        fact = self.ledger.confirm(fact_id, actor)
        self._notify()
        self.reconcile()
        return fact.to_dict()

    def delete_fact(self, fact_id: str, actor: MemoryReviewer) -> bool:
        """Scrub a reviewed fact immediately, even if vector cleanup is delayed."""
        self.initialize()
        deleted = self.ledger.delete_fact(fact_id, actor)
        self._notify()
        self.reconcile()
        return deleted

    def reject_proposal(self, fact_id: str, actor: MemoryReviewer) -> bool:
        """Reject only a proposal, with durable cleanup and no silent downgrade."""
        self.initialize()
        deleted = self.ledger.reject_proposal(fact_id, actor)
        self._notify()
        self.reconcile()
        return deleted

    def delete_conversation(self, conversation_id: str, actor: MemoryReviewer) -> int:
        """Block future derivations before removing current conversation facts."""
        self.initialize()
        deleted = self.ledger.delete_conversation(conversation_id, actor)
        self._notify()
        self.reconcile()
        return deleted

    def reconcile(self) -> dict:
        """Reconcile a bounded batch; a stale upsert cannot acknowledge deletion."""
        self.initialize()
        self.ledger.expire()
        if not self._index_lock.acquire(blocking=False):
            return self.ledger.status()
        try:
            for job in self.ledger.jobs():
                fact = self.ledger.get(job["fact_id"])
                operation = "delete" if fact is None else "upsert"
                try:
                    if operation == "delete":
                        self._collection.delete(ids=[job["fact_id"]])
                    else:
                        self._collection.upsert(ids=[fact.id], documents=[fact.content],
                            metadatas=[{"fact_policy": 1, "revision": fact.revision}])
                    # A concurrent deletion has a newer queued revision. Leave
                    # it pending so the next pass removes this old write.
                    self.ledger.acknowledge(job["fact_id"], job["revision"])
                except Exception as error:
                    self.ledger.acknowledge(job["fact_id"], job["revision"], error=type(error).__name__)
                    logger.warning("Memory index reconciliation failed: %s", type(error).__name__)
            return self.ledger.status()
        finally:
            self._index_lock.release()

    def search_relevant(self, query: str, n_results: int = 5, min_confidence: float = .5) -> list[dict]:
        """Resolve index candidates against live authoritative facts, not documents."""
        if type(n_results) is not int or not 0 < n_results <= 100:
            raise ValueError("invalid_memory_query_limit")
        self.initialize()
        if not self._query_lock.acquire(blocking=False):
            raise RuntimeError("memory_search_busy")
        try:
            count = self._collection.count()
            eligible = self.ledger.eligible_ids(min_confidence)
            if not count or not eligible:
                return []
            results = self._collection.query(query_texts=[query], n_results=min(count, n_results),
                                              ids=eligible, include=["distances"])
            output = []
            ids = results.get("ids") or [[]]
            distances = results.get("distances") or [[]]
            for index, fact_id in enumerate(ids[0]):
                fact = self.ledger.get(fact_id)
                if fact is None or fact.confidence < min_confidence:
                    continue
                distance = distances[0][index] if distances[0] and index < len(distances[0]) else None
                item = fact.to_dict()
                item["similarity"] = 1-distance/2 if type(distance) in (int, float) and math.isfinite(distance) else None
                output.append(item)
            return output
        finally:
            self._query_lock.release()

    def get_by_category(self, category: str, limit: int = 10) -> list[dict]:
        """List live catalog facts without reading unverified index metadata."""
        self.initialize()
        return [fact.to_dict() for fact in self.ledger.list(category=category, limit=limit)]

    def proposals(self, limit: int = 100) -> list[dict]:
        """List only live pending proposals for authenticated review adapters."""
        self.initialize()
        return [fact.to_dict() for fact in self.ledger.list(proposals=True, limit=limit)]

    def get_stats(self) -> dict:
        """Expose lifecycle/cleanup counts, with legacy quarantine explicit."""
        self.initialize()
        return {"total_facts": len(self.ledger.eligible_ids()), "collection": self.index_collection_name,
                "legacy_collection": self._collection_name, "legacy_policy": "quarantined",
                **self.ledger.status()}

    async def _run(self, function, *args, **kwargs):
        with self._resources_lock:
            self._require_open()
            if not self._slots.acquire(blocking=False):
                raise RuntimeError("memory_operation_capacity")
            task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
            self._operations.add(task)

        def finished(completed):
            with self._resources_lock:
                self._operations.discard(completed)
                self._slots.release()
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def astore_fact(self, *args, **kwargs) -> str:
        """Store through an owned worker without blocking capture."""
        return await self._run(self.store_fact, *args, **kwargs)

    async def asearch_relevant(self, *args, **kwargs) -> list[dict]:
        """Search through a bounded owned worker."""
        return await self._run(self.search_relevant, *args, **kwargs)

    async def areconcile(self) -> dict:
        """Run durable cleanup/index maintenance off the event loop."""
        return await self._run(self.reconcile)

    async def acall(self, function, *args, **kwargs):
        """Run another memory producer/reader in this same resource scope."""
        return await self._run(function, *args, **kwargs)

    def resource_status(self) -> dict:
        """Report actual retained work, including cancelled native calls."""
        with self._resources_lock:
            return {"closed": self._closed, "operations": len(self._operations),
                    "index_busy": self._index_lock.locked(), "query_busy": self._query_lock.locked(),
                    "writer_lease": self._lease is not None}

    async def maintain(self) -> None:
        """Run expiry and retry maintenance with tracked, bounded native work."""
        self._maintenance_loop = asyncio.get_running_loop()
        self._maintenance_event = asyncio.Event()
        while not self._closed:
            self._maintenance_event.clear()
            try:
                async with asyncio.timeout(1):
                    await self.areconcile()
                delay = await self.acall(self.ledger.next_maintenance_delay)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("Memory maintenance deferred: %s", type(error).__name__)
                delay = 30
            try:
                async with asyncio.timeout(delay):
                    await self._maintenance_event.wait()
            except TimeoutError:
                pass

    def _close_resources(self):
        with self._index_lock, self._query_lock, self._initializing:
            self.ledger.close()
            if self._lease is not None:
                os.close(self._lease)
                self._lease = None

    async def aclose(self) -> None:
        """Close admission and retain ownership when native cleanup is blocked."""
        with self._resources_lock:
            self._closed = True
        if self._close_task is None:
            async def finish():
                while True:
                    with self._resources_lock:
                        tasks = list(self._operations)
                    if not tasks:
                        break
                    await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.to_thread(self._close_resources)
            self._close_task = asyncio.create_task(finish())
        cancelled = False
        while not self._close_task.done():
            try:
                async with asyncio.timeout(self.shutdown_timeout_s):
                    await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                cancelled = True
            except TimeoutError as error:
                raise MemoryCloseTimeout("memory_resources_still_running") from error
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError
