"""
LiveEventBus — pub/sub in-process para eventos del pipeline de voz hacia el dashboard.

Desacopla el voice_pipeline (productor) del WS del dashboard (consumidor): el pipeline
publica `LiveEvent`s sin saber si alguien escucha; el dashboard se subscribe y reenvía
cada evento por WebSocket a los clientes conectados.

Frame contract (consumido por src/dashboard/frontend/obs/src/app.jsx):
    {"type": "turn"|"alert"|"wake"|"tts"|"cooldown", "payload": {...}, "ts": "HH:MM:SS",
     "event_id": UUID, "expires_at": Unix seconds}
    Retirement controls use type="retire" with payload={"event_ids": [UUID]}.
"""

import asyncio
from contextvars import Context
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Awaitable, Callable

from src.dashboard.live_retention import LiveRetention, RetainedLiveEvent

from src.memory.fact_ledger import MemorySource

logger = logging.getLogger(__name__)


class LiveEventType(StrEnum):
    TURN = "turn"
    ALERT = "alert"
    WAKE = "wake"
    TTS = "tts"
    COOLDOWN = "cooldown"


class OverflowPolicy(StrEnum):
    """Política cuando la cola de un subscriber está llena al `publish()`.

    DROP_OLDEST: descarta el evento más viejo y mete el nuevo. Pipeline nunca
        espera; dashboard pierde lo ya renderizado en bursts.
    DROP_NEWEST: descarta el evento entrante. Pipeline nunca espera; dashboard
        se "congela" en bursts.
    """

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"


@dataclass(frozen=True)
class LiveEvent:
    """Propuesta del productor; el bus valida y crea su propia proyección."""

    type: LiveEventType
    payload: dict = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    sources: tuple[MemorySource, ...] = field(default=(), repr=False)

    def to_frame(self) -> dict:
        return {"type": self.type.value, "payload": self.payload, "ts": self.ts}


Subscriber = Callable[[RetainedLiveEvent], Awaitable[None]]


class LiveEventBus:
    """Pub/sub in-process — `publish()` no bloquea al productor.

    Cada subscriber tiene su propia `asyncio.Queue`. Si la cola se llena, se
    aplica la `OverflowPolicy` configurada. Diseñado para que el voice_pipeline
    (presupuesto <300ms end-to-end) jamás espere por un cliente WS lento.
    """

    def __init__(
        self,
        queue_size: int = 256,
        overflow_policy: OverflowPolicy | str = OverflowPolicy.DROP_OLDEST,
        *, source_policy=None, clock=None, max_subscribers=16, close_timeout_s=1, receipts=None,
    ):
        try:
            self._overflow_policy = OverflowPolicy(overflow_policy)
        except ValueError as e:
            valid = [p.value for p in OverflowPolicy]
            raise ValueError(
                f"overflow_policy={overflow_policy!r} inválido. Válidos: {valid}"
            ) from e
        self._queue_size = queue_size
        self._subscribers: dict[int, asyncio.Queue[RetainedLiveEvent]] = {}
        self._dropped: dict[int, int] = {}
        self._next_id = 0
        self._retention = None
        self._rejected = 0
        if source_policy is not None:
            self._retention = LiveRetention(self._subscribers, source_policy, clock=clock,
                capacity=queue_size, max_subscribers=max_subscribers, close_timeout_s=close_timeout_s, receipts=receipts)

    async def publish(self, event: LiveEvent) -> None:
        """Llamado por el voice_pipeline. Nunca espera lock ni I/O.

        Itera el dict de subscribers haciendo una copia atómica (las ops sobre dict
        en CPython son thread-safe). El dispatch usa solo `put_nowait`/`get_nowait`,
        sin awaits reales.
        """
        if self._retention is None:
            self._rejected += 1
            return
        if self._retention is not None:
            if type(event) is not LiveEvent or type(event.type) is not LiveEventType:
                self._retention.rejected += 1
                return
            event = self._retention.admit(event)
            if event is None:
                return
        for sub_id, q in list(self._subscribers.items()):
            self._dispatch_to_subscriber(sub_id, q, event)
            if self._retention is not None:
                self._retention.notify(sub_id)

    def _dispatch_to_subscriber(
        self, sub_id: int, queue: asyncio.Queue[RetainedLiveEvent], event: RetainedLiveEvent
    ) -> None:
        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        if self._overflow_policy is OverflowPolicy.DROP_OLDEST:
            try:
                queue.get_nowait()
                queue.task_done()
                queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass
        # DROP_NEWEST: simplemente no insertamos.

        n = self._dropped.get(sub_id, 0) + 1
        self._dropped[sub_id] = n
        if n == 1 or n % 100 == 0:
            Context().run(logger.warning, 'Live subscriber %s dropped %s events (policy=%s)',
                sub_id, n, self._overflow_policy.value)

    async def subscribe(self) -> tuple[int, asyncio.Queue[RetainedLiveEvent]]:
        sub_id = self._next_id
        self._next_id += 1
        queue: asyncio.Queue[RetainedLiveEvent] = asyncio.Queue(maxsize=self._queue_size)
        if self._retention is not None:
            self._retention.attach(sub_id, queue)
        self._subscribers[sub_id] = queue
        Context().run(logger.debug, 'Live subscriber %s attached', sub_id)
        return sub_id, queue

    async def unsubscribe(self, sub_id: int) -> None:
        if self._retention is not None:
            await self._retention.detach(sub_id)
        self._subscribers.pop(sub_id, None)
        self._dropped.pop(sub_id, None)
        Context().run(logger.debug, 'Live subscriber %s detached', sub_id)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def dropped_count(self, sub_id: int) -> int:
        return self._dropped.get(sub_id, 0)

    def start(self) -> None:
        """Start source deadline maintenance before publishers are admitted."""
        if self._retention is not None:
            self._retention.start()

    async def stream(self, sub_id, sender, close=None, receiver=None, guard=None):
        """Retain the actual scoped transport after a cancelled observer."""
        if self._retention is not None:
            return await self._retention.stream(sub_id, sender, close, receiver, guard)
        if close is not None:
            await close()

    async def browser_stream(self, sub_id, sender, close=None, receiver=None, guard=None):
        """Require the durable browser protocol on the network-facing live route."""
        if self._retention is None:
            if close is not None:
                await close()
            return
        return await self._retention.stream(sub_id,sender,close,receiver,guard,browser=True)

    def pending_conversation(self, conversation_id: str) -> int:
        """Read local callback ownership for the STREAMS cleanup driver."""
        return self._retention.pending_conversation(conversation_id) if self._retention is not None else 0

    def resource_status(self) -> dict:
        """Read content-free ownership counts."""
        return self._retention.status() if self._retention is not None else {'subscribers':self.subscriber_count,'rejected':self._rejected}

    def begin_close(self) -> asyncio.Task:
        """Seal scoped queues and drain actual socket resources."""
        if self._retention is not None:
            return self._retention.begin_close()
        async def finish():
            self._subscribers.clear()
        return asyncio.create_task(finish())

    async def aclose(self) -> None:
        """Observe closure without cancelling actual scoped sends."""
        if self._retention is not None:
            await self._retention.aclose()
        else:
            await self.begin_close()
