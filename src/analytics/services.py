"""Runtime construction of the private event policy and its owned cleanup."""

from contextlib import AsyncExitStack

from src.analytics.event_logger import EventLogger
from src.memory.fact_ledger import FactLedger


async def build_event_logger(config: dict, *, shutdown: AsyncExitStack,
                             source_policy: FactLedger | None = None, clock=None,
                             io_worker=None) -> EventLogger | None:
    """Register cleanup before creating private tables off-loop."""
    settings = config.get("analytics", {})
    if not settings.get("enabled", True):
        return None
    events = EventLogger(settings.get("events_db", "./data/events.db"),
        retention_days=settings.get("retention_days", 90), source_policy=source_policy,
        clock=clock, io_worker=io_worker,
        observation_timeout_s=settings.get("operation_timeout_s", .1))
    shutdown.push_async_callback(events.aclose)
    await events.acall(events.initialize)
    events.start()
    return events
