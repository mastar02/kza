"""
Rolling metrics tracker para endpoints LLM.

Cada call exitosa registra (timestamp, completion_tokens, elapsed_ms) en una
ventana móvil de 5 minutos por endpoint. El dashboard lee snapshots para
mostrar tok/s promedio y ttft_ms p50.

Diseñado para no bloquear: registro O(1) con deque maxlen, snapshot O(N) sobre
una ventana acotada (~30-60 entries típicamente).
"""

import logging
import statistics
import time
from collections import deque
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


class LLMMetricsTracker:
    """Per-endpoint rolling window. Thread-safe (router puede ser llamado
    desde múltiples tasks)."""

    def __init__(self, window_s: float = 300.0, max_per_endpoint: int = 200):
        self._window_s = window_s
        self._max = max_per_endpoint
        # endpoint_id → deque[(ts, tokens, elapsed_ms)]
        self._data: dict[str, deque] = {}
        self._lock = Lock()

    def record(self, endpoint_id: str, tokens: int, elapsed_ms: float) -> None:
        if elapsed_ms <= 0:
            return
        now = time.time()
        with self._lock:
            q = self._data.get(endpoint_id)
            if q is None:
                q = deque(maxlen=self._max)
                self._data[endpoint_id] = q
            q.append((now, int(tokens), float(elapsed_ms)))

    def snapshot(self, endpoint_id: str) -> Optional[dict]:
        """Devuelve {tps, ttft_ms, calls, last_call_ts} o None si sin datos."""
        cutoff = time.time() - self._window_s
        with self._lock:
            q = self._data.get(endpoint_id)
            if not q:
                return None
            entries = [e for e in q if e[0] >= cutoff]
        if not entries:
            return None
        elapsed = [e[2] for e in entries]
        tokens_total = sum(e[1] for e in entries)
        ms_total = sum(elapsed)
        tps = (tokens_total / (ms_total / 1000)) if ms_total > 0 else None
        return {
            "tps": round(tps, 1) if tps is not None else None,
            "ttft_ms": int(statistics.median(elapsed)),
            "calls": len(entries),
            "last_call_ts": max(e[0] for e in entries),
        }

    def reset(self, endpoint_id: str | None = None) -> None:
        with self._lock:
            if endpoint_id is None:
                self._data.clear()
            else:
                self._data.pop(endpoint_id, None)
