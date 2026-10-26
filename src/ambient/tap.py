"""MultiChannelTap — puente del audio callback (thread C) al ambient path.

push() corre dentro del callback de sounddevice: O(1), sin locks bloqueantes,
sin excepciones hacia afuera. deque(maxlen) es thread-safe para append/popleft
en CPython. Si el consumidor se atrasa, se descarta FIFO (perder transcript
ambiental es aceptable; bloquear el audio del command path no).
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from src.memory.sources import CaptureBoundary

# ~100s de audio por room @ chunks de 80ms. A 6ch float32: ~12 MB/room.
DEFAULT_MAXLEN_CHUNKS = 1250

# (timestamp epoch s del chunk, audio multicanal, TTS propio activo)
_TapEntry = tuple[float, np.ndarray, bool]


class MultiChannelTap:
    """Ring buffer por room de chunks multicanal (ts, chunk, tts_active)."""

    def __init__(self, maxlen_chunks: int = DEFAULT_MAXLEN_CHUNKS, *, clock=None):
        self._maxlen = maxlen_chunks
        self._queues: dict[str, deque[_TapEntry]] = {}
        self._boundaries: dict[str, CaptureBoundary] = {}
        self._clock = clock or time.time
        self._closed = False

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
    ) -> None:
        """Encolar un chunk. Room no registrada = no-op silencioso (fail-open).

        El caller debe pasar ``indata.copy()`` (el array MULTICANAL completo) —
        NO el ``audio_chunk`` monocanal ya recortado del command path. PortAudio
        reutiliza el buffer de ``indata`` al retornar del callback; sin copia
        propia el array encolado queda corrupto. Acá NO se copia de nuevo.
        """
        if self._closed:
            return
        timestamp = ts if ts is not None else self._clock()
        # Snapshot the queue before reading its fence. Publication installs
        # the stricter fence before replacing the queue, so an old callback
        # cannot combine an old boundary with the newly published deque.
        q = self._queues.get(room_id)
        if q is None:
            return
        boundary = self._boundaries.get(room_id)
        if boundary is not None and not boundary.allows(timestamp):
            return
        q.append((timestamp, chunk, tts_active))

    def drain(self, room_id: str) -> list[tuple[float, np.ndarray, bool]]:
        """Vaciar y devolver los chunks pendientes de una room (orden FIFO)."""
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
        if self._closed or previous == boundary:
            return 0
        self._boundaries[room_id] = boundary
        queue = self._queues.get(room_id)
        if queue is None:
            return 0
        items = list(queue)
        retained = deque((item for item in items if boundary.allows(item[0])), maxlen=self._maxlen)
        self._queues[room_id] = retained
        queue.clear()
        return len(items)-len(retained)

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
