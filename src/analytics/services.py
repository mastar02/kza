"""Runtime construction of the private event policy and its owned cleanup."""

from contextlib import AsyncExitStack

from src.analytics.event_logger import EventLogger
from src.memory.fact_ledger import FactLedger
from src.memory.privacy_services import PrivacyServices


async def build_event_logger(config: dict, *, shutdown: AsyncExitStack,
                             source_policy: FactLedger | None = None, clock=None,
                             io_worker=None, privacy: PrivacyServices | None = None, retention=None) -> EventLogger | None:
    """Register cleanup before creating private tables off-loop."""
    settings = config.get("analytics", {})
    if not settings.get("enabled", True):
        return None
    if privacy is not None:
        privacy.require_available()
        if source_policy is not None and source_policy is not privacy.ledger:
            raise ValueError('event_authority_mismatch')
        source_policy = privacy.ledger
    events = EventLogger(settings.get("events_db", "./data/events.db"),
        retention_days=settings.get("retention_days", 90), source_policy=source_policy,
        clock=clock, io_worker=io_worker,
        retention_notify=retention.notify if retention is not None else None,
        observation_timeout_s=settings.get("operation_timeout_s", .1))
    shutdown.push_async_callback(events.aclose)
    if privacy is not None:
        privacy.add_borrower(events.begin_close)
    await events.acall(events.initialize)
    if retention is None:
        events.start()
    return events
