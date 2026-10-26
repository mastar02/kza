"""Capture scopes describe recorded exchanges, independently of speaker identity."""

from dataclasses import replace
import threading
from uuid import uuid4

from src.memory.fact_ledger import FactLedger, MemoryOrigin, MemorySource


class CaptureScopes:
    """Share room/origin scopes across all active capture producers."""

    def __init__(self, ledger: FactLedger, *, admission_started_at: float | None = None):
        self.ledger = ledger
        self._admission_started_at = admission_started_at
        self._scopes = {}
        self._revocation_fences = {}
        self._lock = threading.Lock()

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
                previous = self._scopes.get(key)
                revoked_at = self.ledger.conversation_deleted_at(previous[0]) if previous is not None else None
                if revoked_at is not None:
                    self._revocation_fences[key] = max(revoked_at, self._revocation_fences.get(key, revoked_at))
                fence = self._revocation_fences.get(key)
                if fence is not None and captured_at <= fence:
                    raise PermissionError("capture_precedes_conversation_revocation")
                if previous is None or self.ledger.conversation_revoked(previous[0]) or not 0 <= captured_at-previous[1] <= 120 or captured_at-previous[2] > 3600:
                    conversation_id, started = source.conversation_id, captured_at
                else:
                    conversation_id, started = previous[0], previous[2]
                self._scopes[key] = (conversation_id, captured_at, started)
        return replace(source, conversation_id=conversation_id) if source.conversation_id != conversation_id else source
