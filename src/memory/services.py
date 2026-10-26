"""Application construction and lifecycle for policy-owned memory."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass

from src.memory.memory_manager import MemoryManager


@dataclass
class MemoryServices:
    """Share one manager, its writer lease and its actual maintenance task."""

    manager: MemoryManager
    maintenance: asyncio.Task | None = None

    def start(self) -> None:
        """Start exactly one owned maintenance loop."""
        if self.maintenance is None:
            self.maintenance = asyncio.create_task(self.manager.long_term.maintain())

    async def aclose(self) -> None:
        """Stop producers before waiting retained native memory resources."""
        if self.maintenance is not None:
            self.maintenance.cancel()
            await asyncio.gather(self.maintenance, return_exceptions=True)
        await self.manager.long_term.aclose()


async def build_memory_services(config: dict, *, shutdown: AsyncExitStack, chroma_client=None,
                                clock=None, privacy=None) -> MemoryServices | None:
    """Construct the runtime graph off-loop and register cleanup before init."""
    settings = config.get("memory", {})
    if not settings.get("enabled", True):
        return None
    if privacy is not None:
        privacy.require_available()

    def construct():
        client = chroma_client
        if client is None:
            import chromadb
            client = chromadb.PersistentClient(path=settings.get("chroma_path", "./data/memory_db"))
        return MemoryManager(client,
            preferences_path=settings.get("preferences_path", "./data/preferences.json"),
            max_short_term_turns=settings.get("short_term_size", 10),
            collection_name=settings.get("long_term_collection", "user_memories"),
            ledger_path=settings.get("privacy_db_path", "./data/memory_privacy.db"),
            automatic_days=settings.get("automatic_retention_days", 30), clock=clock, privacy=privacy)

    building = asyncio.create_task(asyncio.to_thread(construct))
    cancelled = False
    while not building.done():
        try:
            await asyncio.shield(building)
        except asyncio.CancelledError:
            cancelled = True
    services = MemoryServices(building.result())
    shutdown.push_async_callback(services.aclose)
    if cancelled:
        raise asyncio.CancelledError
    await services.manager.ainitialize()
    services.start()
    return services
