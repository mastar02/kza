"""Capture scopes describe recorded exchanges, independently of speaker identity."""

from dataclasses import dataclass, replace
import math
import threading
from uuid import uuid4

from src.memory.fact_ledger import FactLedger, MemoryOrigin, MemorySource


@dataclass(frozen=True)
class CaptureBoundary:
    """Admission and transient retirement dates, without private capture values."""

    started_at: float | None
    retired_through: float

    def tighten(self, other: "CaptureBoundary") -> "CaptureBoundary":
        """Keep the strictest published dates across delayed observations."""
        starts = [v for v in (self.started_at, other.started_at) if v is not None]
        return CaptureBoundary(max(starts) if starts else None,
                               max(self.retired_through, other.retired_through))

    def allows(self, timestamp: float) -> bool:
        """Reject nonfinite, pre-start, expired and fenced transient audio."""
        return (type(timestamp) in (int, float) and math.isfinite(timestamp)
                and (self.started_at is None or timestamp >= self.started_at)
                and timestamp > self.retired_through)


class CaptureScopes:
    """Share room/origin scopes across all active capture producers."""

    def __init__(self, ledger: FactLedger, *, admission_started_at: float | None = None):
        self.ledger = ledger
        self._admission_started_at = admission_started_at
        self._scopes = {}
        self._revocation_fences = {}
        self._buffer_expiry_floor = float('-inf')
        self._lock = threading.Lock()

    def buffer_boundary(self, room_id: str | None, *, origin: MemoryOrigin = MemoryOrigin.AMBIENT) -> CaptureBoundary:
        """Publish the same room fence used by source admission for transient buffers.

        Pending revocations may discard transient inputs, like private caches.
        They never authorize deletion from persistent capture stores.
        """
        key = (origin, room_id)
        with self._lock:
            self._observe_revocation(key)
            self._buffer_expiry_floor = max(self._buffer_expiry_floor, self.ledger.clock()-48*3600)
            return CaptureBoundary(self._admission_started_at,
                max(self._buffer_expiry_floor, self._revocation_fences.get(key, float('-inf'))))

    def _observe_revocation(self, key):
        previous = self._scopes.get(key)
        revoked_at = self.ledger.conversation_deleted_at(previous[0]) if previous is not None else None
        if revoked_at is not None:
            self._revocation_fences[key] = max(revoked_at, self._revocation_fences.get(key, revoked_at))

    def source_for_capture(self, capture_id: str, captured_at: float, room_id: str | None,
                           *, conversation_id: str | None = None, expires_at: float | None = None,
                           origin: MemoryOrigin = MemoryOrigin.VOICE) -> MemorySource:
        """Keep explicit IDs; otherwise split at 120s idle, one hour or revocation."""
        source = MemorySource(capture_id, conversation_id or str(uuid4()), captured_at,
            expires_at if expires_at is not None else min(captured_at, self.ledger.clock())+48*3600, origin)
        if conversation_id is None:
            if self._admission_started_at is not None and captured_at < self._admission_started_at:
                raise PermissionError("capture_precedes_privacy_start")
            key = (origin, room_id)
            with self._lock:
                if captured_at <= self._buffer_expiry_floor:
                    raise PermissionError("capture_precedes_buffer_retirement")
                previous = self._scopes.get(key)
                self._observe_revocation(key)
                fence = self._revocation_fences.get(key)
                if fence is not None and captured_at <= fence:
                    raise PermissionError("capture_precedes_conversation_revocation")
                if previous is None or self.ledger.conversation_revoked(previous[0]) or not 0 <= captured_at-previous[1] <= 120 or captured_at-previous[2] > 3600:
                    conversation_id, started = source.conversation_id, captured_at
                else:
                    conversation_id, started = previous[0], previous[2]
                self._scopes[key] = (conversation_id, captured_at, started)
        return replace(source, conversation_id=conversation_id) if source.conversation_id != conversation_id else source
