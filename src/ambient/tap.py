"""MultiChannelTap — puente del audio callback (thread C) al ambient path.

push() corre dentro del callback de sounddevice: O(1), sin locks bloqueantes,
sin excepciones hacia afuera. deque(maxlen) es thread-safe para append/popleft
en CPython. Si el consumidor se atrasa, se descarta FIFO (perder transcript
ambiental es aceptable; bloquear el audio del command path no).
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from weakref import WeakMethod, WeakSet

import numpy as np

from src.memory.sources import CaptureBoundary, CaptureEpoch
from src.audio.input_callbacks import InputCallbacks

# ~100s de audio por room @ chunks de 80ms. A 6ch float32: ~12 MB/room.
DEFAULT_MAXLEN_CHUNKS = 1250

# (timestamp epoch s del chunk, audio multicanal, TTS propio activo)
@dataclass(frozen=True)
class TapPacket:
    """Copied input with the metadata admission that preceded clocks and arrays."""

    timestamp: float
    audio: np.ndarray = field(repr=False)
    tts_active: bool
    epoch: CaptureEpoch | None = field(default=None, repr=False)


_PUSH_BOUNDARY = CaptureBoundary(None, float('-inf'))
_UNSTAMPED = object()


class MultiChannelTap:
    """Ring buffer por room de chunks multicanal (ts, chunk, tts_active)."""

    def __init__(self, maxlen_chunks: int = DEFAULT_MAXLEN_CHUNKS, *, clock=None):
        self._maxlen = maxlen_chunks
        self._queues: dict[str, deque[TapPacket]] = {}
        self._boundaries: dict[str, CaptureBoundary] = {}
        self._clock = clock or time.time
        self._closed = False
        self._paused = False
        self._ingress_token = object()
        self._ingress: WeakSet[InputCallbacks] = WeakSet()
        self._pushes = InputCallbacks()
        self._closing = None
        self._capture_ingress = {}
        self._published_epochs = {}
        self._epoch_check = None

    def bind_ingress(self, scopes, room_ids: list[str]) -> None:
        """Install stable control-owned references before exposing microphone input."""
        from src.memory.fact_ledger import MemoryOrigin
        for room_id in room_ids:
            ingress = scopes.ingress(room_id, origin=MemoryOrigin.AMBIENT)
            previous = self._capture_ingress.get(room_id)
            if previous is not None and previous.current.authority is not ingress.current.authority:
                raise ValueError('ambient_ingress_authority_mismatch')
            self._capture_ingress[room_id] = ingress
        self._epoch_check = WeakMethod(scopes.epoch_available)

    def input_epoch(self, room_id: str) -> CaptureEpoch | None:
        """Read admission metadata without clocks, locks or authority queries."""
        ingress = self._capture_ingress.get(room_id)
        return ingress.current if ingress is not None else None

    @property
    def ingress_token(self) -> object:
        """Metadata identity for the callback's independent ambient branch."""
        return self._ingress_token

    @property
    def accepting(self) -> bool:
        """Let voice callbacks skip the disabled ambient branch before copying."""
        return not self._closed and not self._paused

    def pause(self) -> None:
        """Seal ingress and detach copied queues without closing command capture."""
        self._paused = True
        for room_id,queue in tuple(self._queues.items()):
            self._queues[room_id] = deque(maxlen=self._maxlen)
            queue.clear()

    def resume(self) -> None:
        """Reopen only after the control owner has retired prior ingress epochs."""
        if self._closed or self.ingress_pending():
            raise RuntimeError('ambient_ingress_still_owned')
        self._paused = False

    def register_ingress(self, callbacks: InputCallbacks) -> None:
        """Register metadata ownership before exposing this tap to a microphone."""
        self._ingress.add(callbacks)

    def ingress_pending(self) -> bool:
        """Unattributed copies remain pending independently of the voice source."""
        return self._pushes.status()['active'] > 0 or any(
            callbacks.branch_pending(self._ingress_token) for callbacks in tuple(self._ingress))

    def input_boundary(self, room_id: str) -> CaptureBoundary:
        """Snapshot control-loop metadata before a microphone touches its input."""
        return self._boundaries.get(room_id, _PUSH_BOUNDARY)

    def accepts(self, room_id: str, timestamp: float, boundary: CaptureBoundary, *, epoch: object = _UNSTAMPED) -> bool:
        """Reject a copy admitted before a published ambient boundary changed."""
        current = self.input_boundary(room_id)
        epoch = self.input_epoch(room_id) if epoch is _UNSTAMPED else epoch
        admitted = room_id not in self._capture_ingress or epoch is self.input_epoch(room_id) and not epoch.retired
        return self.accepting and admitted and current == boundary and current.allows(timestamp)

    def register_room(self, room_id: str) -> None:
        """Registrar una room (idempotente)."""
        if not self._closed and room_id not in self._queues:
            self._queues[room_id] = deque(maxlen=self._maxlen)

    def push(
        self,
        room_id: str,
        chunk: np.ndarray,
        ts: float | None = None,
        tts_active: bool = False,
        *,
        boundary: CaptureBoundary | None = None,
        epoch: object = _UNSTAMPED,
    ) -> None:
        """Encolar un chunk. Room no registrada = no-op silencioso (fail-open).

        El caller debe pasar ``indata.copy()`` (el array MULTICANAL completo) —
        NO el ``audio_chunk`` monocanal ya recortado del command path. PortAudio
        reutiliza el buffer de ``indata`` al retornar del callback; sin copia
        propia el array encolado queda corrupto. Acá NO se copia de nuevo.
        """
        epoch = self.input_epoch(room_id) if epoch is _UNSTAMPED else epoch
        ticket = self._pushes.enter(_PUSH_BOUNDARY)
        q = None
        try:
            if ticket is None or not self.accepting:
                return
            timestamp = ts if ts is not None else self._clock()
            # Snapshot the queue before reading its fence. Publication installs
            # the stricter fence before replacing the queue, so an old callback
            # cannot combine an old boundary with the newly published deque.
            q = self._queues.get(room_id)
            if q is None:
                return
            current = self.input_boundary(room_id)
            if (boundary is not None and boundary != current) or not current.allows(timestamp):
                return
            if room_id in self._capture_ingress and (epoch is not self.input_epoch(room_id) or epoch.retired):
                return
            q.append(TapPacket(timestamp, chunk, tts_active, epoch))
        finally:
            chunk = q = None
            if ticket is not None:
                self._pushes.leave(ticket)
            ticket = None

    def drain(self, room_id: str) -> list[tuple[float, np.ndarray, bool]]:
        """Vaciar y devolver los chunks pendientes de una room (orden FIFO)."""
        return [(item.timestamp, item.audio, item.tts_active) for item in self.drain_owned(room_id)]

    def drain_owned(self, room_id: str) -> list[TapPacket]:
        """Transfer copied input together with its original metadata admission."""
        q = self._queues.get(room_id)
        if q is None:
            return []
        items = []
        while True:
            try:
                items.append(q.popleft())
            except IndexError:
                return items

    def publish_boundary(self, room_id: str, boundary: CaptureBoundary) -> int:
        """Detach old queues and publish admission metadata without blocking push.

        A callback already holding the old deque may lose its chunk. It cannot
        append to the new queue or resurrect a retired one.
        """
        previous = self._boundaries.get(room_id)
        if previous is not None:
            boundary = previous.tighten(boundary)
        if self._closed:
            return 0
        epoch = self.input_epoch(room_id)
        changed = previous != boundary or epoch is not self._published_epochs.get(room_id)
        self._boundaries[room_id] = boundary
        self._published_epochs[room_id] = epoch
        queue = self._queues.get(room_id)
        if queue is None:
            return 0
        items = list(queue)
        retained = deque((item for item in items if boundary.allows(item.timestamp)
                          and self._packet_available(room_id, item)), maxlen=self._maxlen)
        if changed or len(retained) != len(items):
            self._queues[room_id] = retained
            queue.clear()
        return len(items)-len(retained)

    def _packet_available(self, room_id, item):
        if room_id not in self._capture_ingress:
            return True
        from src.memory.fact_ledger import MemoryOrigin
        check = self._epoch_check() if self._epoch_check is not None else None
        return check is not None and check(item.epoch, room_id, origin=MemoryOrigin.AMBIENT)

    def close(self) -> None:
        """Close admission and detach buffered audio, including late callbacks."""
        self._closed = True
        queues, self._queues = self._queues, {}
        self._boundaries.clear()
        for queue in queues.values():
            queue.clear()

    def buffered_count(self) -> int:
        """Count retained chunks without projecting audio or private timestamps."""
        return sum(map(len, self._queues.values()))

    def begin_close(self) -> asyncio.Task:
        """Seal queues and wait for actual copies and pushes, without closing voice."""
        self.close()
        if self._closing is None:
            pushes = self._pushes.begin_close()

            async def finish():
                await pushes
                while self.ingress_pending():
                    await asyncio.sleep(.005)

            self._closing = asyncio.create_task(finish())
        return self._closing
