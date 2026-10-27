"""Durable Live browser receipts and local cleanup without certifying other views."""

from src.dashboard.live_event_bus import LiveEventBus
from src.dashboard.live_receipts import LiveReceiptStore
from src.memory.erasure_journal import ErasureParticipant, ErasureResult, ErasureStep

LIVE_TARGET = {'kind':'live_browser_receipts','protocol':1}


class LiveStreamEraser:
    """A live ticket needs a browser ACK and an exited actual local callback."""

    def __init__(self, privacy, bus, receipts):
        self.privacy, self.bus, self.receipts = privacy,bus,receipts

    async def clean(self, step: ErasureStep) -> ErasureResult:
        """Return progress without holding a coordinator slot for offline pages."""
        if (step.participant is not ErasureParticipant.STREAMS or step.version != 1
                or step.descriptor != LIVE_TARGET):
            raise ValueError('unsupported_live_stream_target')
        self.privacy.require_available()
        targets = await self.privacy.acall(self.privacy.ledger.erasures.targets,ErasureParticipant.STREAMS)
        if not any(row['id']==step.target_id and row['version']==1 and row['descriptor']==LIVE_TARGET for row in targets):
            raise ValueError('unregistered_live_stream_target')
        if not self.privacy.ledger.conversation_committed_revoked(step.conversation_id):
            raise RuntimeError('live_erasure_requires_committed_authority')
        self.bus.pending_conversation(step.conversation_id)
        pending = await self.receipts.pending(step.conversation_id)
        pending += self.bus.pending_conversation(step.conversation_id)
        return ErasureResult(pending==0,pending_count=pending)


async def build_live_services(config: dict, *, privacy, owner, shutdown) -> LiveEventBus:
    """Register the owner before metadata work or private publishers can start."""
    receipts = LiveReceiptStore(privacy.ledger)
    settings = config.get('dashboard',{})
    bus = LiveEventBus(queue_size=settings.get('live_event_queue_size',256),
        overflow_policy=settings.get('live_event_overflow_policy','drop_oldest'),
        source_policy=privacy.ledger,clock=privacy.ledger.clock,receipts=receipts)
    privacy.add_borrower(bus.begin_close)
    shutdown.push_async_callback(bus.aclose)
    driver = LiveStreamEraser(privacy,bus,receipts)
    owner.register(ErasureParticipant.STREAMS,driver.clean,versions=frozenset({1}),begin_close=bus.begin_close)
    await privacy.acall(privacy.ledger.erasures.register_target,ErasureParticipant.STREAMS,1,LIVE_TARGET)
    await privacy.acall(privacy.ledger.erasures.register_target,ErasureParticipant.STREAMS,2,
        {'kind':'unresolved_stream_inventory'})
    bus.start()
    return bus
