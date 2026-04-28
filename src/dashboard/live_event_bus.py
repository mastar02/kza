"""
LiveEventBus — pub/sub in-process para eventos del pipeline de voz hacia el dashboard.

Desacopla el voice_pipeline (productor) del WS del dashboard (consumidor): el pipeline
publica `LiveEvent`s sin saber si alguien escucha; el dashboard se subscribe y reenvía
cada evento por WebSocket a los clientes conectados.

Frame contract (matchea README del prototipo):
    {"type": "turn"|"alert"|"wake"|"tts"|"cooldown", "payload": {...}, "ts": "HH:MM:SS"}
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


class LiveEventType(StrEnum):
    TURN = "turn"
    ALERT = "alert"
    WAKE = "wake"
    TTS = "tts"
    COOLDOWN = "cooldown"


@dataclass
class LiveEvent:
    type: LiveEventType
    payload: dict = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def to_frame(self) -> dict:
        return {"type": self.type.value, "payload": self.payload, "ts": self.ts}


Subscriber = Callable[[LiveEvent], Awaitable[None]]


class LiveEventBus:
    """
    Bus in-process. Productores llaman `publish()`; consumidores `subscribe()`.

    Diseñado para no bloquear al productor: cada subscriber recibe via su propia
    asyncio.Queue. Si la queue se llena, se aplica la `OverflowPolicy` configurada.
    """

    def __init__(self, queue_size: int = 256, overflow_policy: str = "drop_oldest"):
        # TODO(user-contribution): ver `_dispatch_to_subscriber` más abajo.
        self._queue_size = queue_size
        self._overflow_policy = overflow_policy
        self._subscribers: dict[int, asyncio.Queue[LiveEvent]] = {}
        self._next_id = 0
        self._lock = asyncio.Lock()

    async def publish(self, event: LiveEvent) -> None:
        """Llamado por el voice_pipeline. NUNCA debe bloquear más de unos micros."""
        async with self._lock:
            queues = list(self._subscribers.values())
        for q in queues:
            await self._dispatch_to_subscriber(q, event)

    async def _dispatch_to_subscriber(
        self, queue: asyncio.Queue[LiveEvent], event: LiveEvent
    ) -> None:
        """Aplica overflow policy. Nunca bloquea al productor."""
        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        policy = self._overflow_policy
        if policy == "drop_oldest":
            try:
                queue.get_nowait()
                queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass
        elif policy == "drop_newest":
            return
        else:
            logger.warning(f"[LiveEventBus] unknown overflow_policy={policy!r}, dropping")

    async def subscribe(self) -> tuple[int, asyncio.Queue[LiveEvent]]:
        """Devuelve (id, queue). El consumidor hace `await queue.get()` en loop."""
        async with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            queue: asyncio.Queue[LiveEvent] = asyncio.Queue(maxsize=self._queue_size)
            self._subscribers[sub_id] = queue
        logger.debug(f"[LiveEventBus] subscriber {sub_id} attached")
        return sub_id, queue

    async def unsubscribe(self, sub_id: int) -> None:
        async with self._lock:
            self._subscribers.pop(sub_id, None)
        logger.debug(f"[LiveEventBus] subscriber {sub_id} detached")

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
