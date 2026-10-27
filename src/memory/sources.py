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


@dataclass(eq=False, repr=False)
class CaptureEpoch:
    """Metadata admission shared by inputs predating the first native source."""

    room_id: str | None
    origin: MemoryOrigin
    expires_at: float
    conversation_id: str | None = None
    retired: bool = False
    authority: object | None = None


@dataclass(repr=False)
class CaptureIngress:
    """Stable control-owned reference; callbacks only snapshot its current epoch."""

    current: CaptureEpoch


class CaptureScopes:
    """Share room/origin scopes across all active capture producers."""

    def __init__(self, ledger: FactLedger, *, admission_started_at: float | None = None):
        self.ledger = ledger
        self._admission_started_at = admission_started_at
        self._scopes = {}
        self._revocation_fences = {}
        self._buffer_expiry_floor = float('-inf')
        self._ingress = {}
        self._ingress_authority = object()
        self._lock = threading.Lock()
        ledger.add_revocation_observer(self._retire_conversations)

    def _retire_conversations(self, conversations):
        # Called on the authority worker, never on an input callback. Retire
        # before swapping: a racing reader may lose input, never regain consent.
        with self._lock:
            for key, ingress in self._ingress.items():
                if ingress.current.conversation_id in conversations:
                    ingress.current.retired = True
                    ingress.current = self._new_epoch(key)

    def _new_epoch(self, key):
        return CaptureEpoch(key[1], key[0], self.ledger.clock()+48*3600, authority=self._ingress_authority)

    def _ingress_locked(self, key):
        if key not in self._ingress:
            epoch = self._new_epoch(key)
            previous = self._scopes.get(key)
            if previous is not None:
                epoch.conversation_id = previous[0]
                epoch.expires_at = min(epoch.expires_at, previous[2]+48*3600)
            self._ingress[key] = CaptureIngress(epoch)
        return self._ingress[key]

    def ingress(self, room_id: str | None, *, origin: MemoryOrigin) -> CaptureIngress:
        """Register a room on the control path, before exposing its input callback."""
        with self._lock:
            self._observe_revocation((origin, room_id))
            return self._ingress_locked((origin, room_id))

    def epoch_available(self, epoch: object, room_id: str | None, *, origin: MemoryOrigin) -> bool:
        """Validate native admission against authority, independently of publication."""
        return (self.ledger.authority_available and isinstance(epoch, CaptureEpoch)
                and epoch.authority is self._ingress_authority
                and epoch.room_id == room_id and epoch.origin == origin and not epoch.retired
                and epoch.expires_at > self.ledger.clock()
                and (epoch.conversation_id is None or not self.ledger.conversation_revoked(epoch.conversation_id)))

    def _bind_epoch(self, key, source):
        ingress = self._ingress_locked(key)
        epoch = ingress.current
        if epoch.conversation_id is not None and epoch.conversation_id != source.conversation_id:
            epoch.retired = True
            ingress.current = epoch = self._new_epoch(key)
        epoch.expires_at = min(epoch.expires_at, source.expires_at)
        epoch.conversation_id = source.conversation_id

    def buffer_boundary(self, room_id: str | None, *, origin: MemoryOrigin = MemoryOrigin.AMBIENT) -> CaptureBoundary:
        """Publish the same room fence used by source admission for transient buffers.

        Pending revocations may discard transient inputs, like private caches.
        They never authorize deletion from persistent capture stores.
        """
        key = (origin, room_id)
        with self._lock:
            self._observe_revocation(key)
            self._buffer_expiry_floor = max(self._buffer_expiry_floor, self.ledger.clock()-48*3600)
            starts = [value for value in (self._admission_started_at,) if value is not None]
            if key in self._scopes:
                # Raw input older than the current exchange must not acquire
                # a different conversation ID when it eventually reaches STT.
                starts.append(self._scopes[key][2])
            return CaptureBoundary(max(starts, default=None),
                max(self._buffer_expiry_floor, self._revocation_fences.get(key, float('-inf'))))

    def _observe_revocation(self, key):
        previous = self._scopes.get(key)
        revoked_at = self.ledger.conversation_deleted_at(previous[0]) if previous is not None else None
        if revoked_at is not None:
            self._revocation_fences[key] = max(revoked_at, self._revocation_fences.get(key, revoked_at))
        ingress = self._ingress.get(key)
        if ingress is not None:
            epoch = ingress.current
            if epoch.expires_at <= self.ledger.clock() or (
                epoch.conversation_id is not None and self.ledger.conversation_revoked(epoch.conversation_id)
            ):
                epoch.retired = True
                ingress.current = self._new_epoch(key)

    def source_for_capture(self, capture_id: str, captured_at: float, room_id: str | None,
                           *, conversation_id: str | None = None, expires_at: float | None = None,
                           origin: MemoryOrigin = MemoryOrigin.VOICE) -> MemorySource:
        """Split scopes at 120s idle, one hour or revocation; never move backwards.

        Delayed input within the current exchange keeps its original identity.
        Older raw input is retired; explicit lineage retains its existing ID.
        """
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
                if previous is not None and captured_at < previous[2]:
                    raise PermissionError('capture_precedes_current_scope')
                if previous is None or self.ledger.conversation_revoked(previous[0]) or captured_at-previous[1] > 120 or captured_at-previous[2] > 3600:
                    conversation_id, started = source.conversation_id, captured_at
                    latest = captured_at
                else:
                    conversation_id, started = previous[0], previous[2]
                    latest = max(captured_at, previous[1])
                self._scopes[key] = (conversation_id, latest, started)
                self._bind_epoch(key, replace(source, conversation_id=conversation_id))
        return replace(source, conversation_id=conversation_id) if source.conversation_id != conversation_id else source
