"""Bounded in-memory derived values with immutable capture deadlines."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
import threading
import time
from typing import Generic, TypeVar

from src.memory.fact_ledger import FactLedger, MemorySource, sources_current

T = TypeVar("T")


@dataclass(frozen=True)
class PrivateEntry(Generic[T]):
    """A derived value cannot outlive its original source captures."""

    value: T
    sources: tuple[MemorySource, ...]
    expires_at: float


class TransientCache(Mapping[str, T]):
    """Keep source links and enforce expiry/revocation on every read or write."""

    def __init__(self, max_size: int, *, source_policy: FactLedger | None = None, clock=None):
        if type(max_size) is not int or max_size <= 0:
            raise ValueError("invalid_transient_cache_size")
        self.max_size, self.source_policy = max_size, source_policy
        self.clock = clock or (source_policy.clock if source_policy is not None else time.time)
        self._entries: dict[str, PrivateEntry[T]] = {}
        self._lock = threading.RLock()
        self._closed = False

    def _live(self, sources: tuple[MemorySource, ...]) -> bool:
        return sources_current(sources, self.clock()) and (self.source_policy is None or self.source_policy.sources_available(sources))

    def put(self, key: str, value: T, sources: tuple[MemorySource, ...]) -> bool:
        """Store a copy with its source deadline; cache hits never renew it."""
        with self._lock:
            self.prune()
            if self._closed or not self._live(sources):
                return False
            if key not in self._entries and len(self._entries) >= self.max_size:
                del self._entries[next(iter(self._entries))]
            self._entries[key] = PrivateEntry(deepcopy(value), sources, min(s.expires_at for s in sources))
            return True

    def prune(self) -> int:
        """Remove expired or revoked keys and their values without I/O."""
        with self._lock:
            keys = [key for key, entry in self._entries.items() if not self._live(entry.sources) or entry.expires_at <= self.clock()]
            for key in keys:
                del self._entries[key]
            return len(keys)

    def clear(self) -> None:
        """Discard every retained key and value on shutdown."""
        with self._lock:
            self._entries.clear()

    def close(self) -> None:
        """Close admission before removing data, so late writers cannot refill it."""
        with self._lock:
            self._closed = True
            self._entries.clear()

    def __getitem__(self, key: str) -> T:
        with self._lock:
            self.prune()
            return deepcopy(self._entries[key].value)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            self.prune()
            return iter(tuple(self._entries))

    def __len__(self) -> int:
        with self._lock:
            self.prune()
            return len(self._entries)
