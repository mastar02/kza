"""
LiveEventBus — pub/sub in-process para eventos del pipeline de voz hacia el dashboard.

Desacopla el voice_pipeline (productor) del WS del dashboard (consumidor): el pipeline
publica `LiveEvent`s sin saber si alguien escucha; el dashboard se subscribe y reenvía
cada evento por WebSocket a los clientes conectados.

Frame contract (consumido por src/dashboard/frontend/obs/src/app.jsx):
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
    """Frame inmutable: se broadcastea a N subscribers, mutación corrompería a otros."""

    type: LiveEventType
    payload: dict = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def to_frame(self) -> dict:
        return {"type": self.type.value, "payload": self.payload, "ts": self.ts}


Subscriber = Callable[[LiveEvent], Awaitable[None]]


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
    ):
        try:
            self._overflow_policy = OverflowPolicy(overflow_policy)
        except ValueError as e:
            valid = [p.value for p in OverflowPolicy]
            raise ValueError(
                f"overflow_policy={overflow_policy!r} inválido. Válidos: {valid}"
            ) from e
        self._queue_size = queue_size
        self._subscribers: dict[int, asyncio.Queue[LiveEvent]] = {}
        self._dropped: dict[int, int] = {}
        self._next_id = 0

    async def publish(self, event: LiveEvent) -> None:
        """Llamado por el voice_pipeline. Nunca espera lock ni I/O.

        Itera el dict de subscribers haciendo una copia atómica (las ops sobre dict
        en CPython son thread-safe). El dispatch usa solo `put_nowait`/`get_nowait`,
        sin awaits reales.
        """
        for sub_id, q in list(self._subscribers.items()):
            self._dispatch_to_subscriber(sub_id, q, event)

    def _dispatch_to_subscriber(
        self, sub_id: int, queue: asyncio.Queue[LiveEvent], event: LiveEvent
    ) -> None:
        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        if self._overflow_policy is OverflowPolicy.DROP_OLDEST:
            try:
                queue.get_nowait()
                queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass
        # DROP_NEWEST: simplemente no insertamos.

        n = self._dropped.get(sub_id, 0) + 1
        self._dropped[sub_id] = n
        if n == 1 or n % 100 == 0:
            logger.warning(
                f"[LiveEventBus] subscriber {sub_id} dropped {n} events "
                f"(policy={self._overflow_policy.value})"
            )

    async def subscribe(self) -> tuple[int, asyncio.Queue[LiveEvent]]:
        sub_id = self._next_id
        self._next_id += 1
        queue: asyncio.Queue[LiveEvent] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers[sub_id] = queue
        logger.debug(f"[LiveEventBus] subscriber {sub_id} attached")
        return sub_id, queue

    async def unsubscribe(self, sub_id: int) -> None:
        self._subscribers.pop(sub_id, None)
        self._dropped.pop(sub_id, None)
        logger.debug(f"[LiveEventBus] subscriber {sub_id} detached")

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def dropped_count(self, sub_id: int) -> int:
        return self._dropped.get(sub_id, 0)
