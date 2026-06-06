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

# ~100s de audio por room @ chunks de 80ms. A 6ch float32: ~12 MB/room.
DEFAULT_MAXLEN_CHUNKS = 1250


class MultiChannelTap:
    """Ring buffer por room de chunks multicanal (ts, chunk, tts_active)."""

    def __init__(self, maxlen_chunks: int = DEFAULT_MAXLEN_CHUNKS):
        self._maxlen = maxlen_chunks
        self._queues: dict[str, deque] = {}

    def register_room(self, room_id: str) -> None:
        """Registrar una room (idempotente)."""
        if room_id not in self._queues:
            self._queues[room_id] = deque(maxlen=self._maxlen)

    def push(
        self,
        room_id: str,
        chunk: np.ndarray,
        ts: float | None = None,
        tts_active: bool = False,
    ) -> None:
        """Encolar un chunk. Room no registrada = no-op silencioso (fail-open).

        El caller (audio callback) ya hace .copy() del buffer de PortAudio —
        acá NO se copia de nuevo.
        """
        q = self._queues.get(room_id)
        if q is None:
            return
        q.append((ts if ts is not None else time.time(), chunk, tts_active))

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
