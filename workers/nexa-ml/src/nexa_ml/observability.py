"""Payload-free inference timing shared by RPCs and their model threads."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class InferenceTiming:
    """Measure actual model work independently of queueing and RPC streaming."""

    slot: str = "unresolved"
    voice: str | None = None
    _interval: tuple[int, int | None] | None = field(default=None, init=False, repr=False)

    def wrap[M, T](self, slot: str, call: Callable[[M], T]) -> Callable[[M], T]:
        """Bind the selected slot and time work when it reaches the model lane."""
        self.slot = slot

        def measured(model: M) -> T:
            start = time.perf_counter_ns()
            self._interval = (start, None)
            try:
                return call(model)
            finally:
                self._interval = (start, time.perf_counter_ns())

        return measured

    def snapshot(self) -> tuple[int | None, bool]:
        """Return elapsed inference milliseconds and whether the model finished."""
        interval = self._interval
        if interval is None:
            return None, False
        start, end = interval
        now = time.perf_counter_ns() if end is None else end
        return (now - start) // 1_000_000, end is not None
