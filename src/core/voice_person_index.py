"""Content-free relationship admission shared by every view of one fact ledger."""

from dataclasses import dataclass, field
import threading


@dataclass(frozen=True)
class VoicePersonIndexSnapshot:
    available: bool
    users: frozenset[str] = field(repr=False)


class VoicePersonIndex:
    """Publish one immutable proof for all native helpers on the same ledger."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pending = 0
        self._uncertain = True
        self._known: frozenset[str] = frozenset()
        self.snapshot = VoicePersonIndexSnapshot(False, frozenset())

    def invalidate(self) -> None:
        """Withdraw prior SQL knowledge before closing or reopening its source."""
        with self._lock:
            self._uncertain = True
            self.snapshot = VoicePersonIndexSnapshot(False, self._known)

    def begin(self, *, withdraw: bool = True) -> None:
        if type(withdraw) is not bool:
            raise ValueError('invalid_voice_person_admission')
        with self._lock:
            self.pending += 1
            if withdraw:
                self.snapshot = VoicePersonIndexSnapshot(False, self._known)

    def finish(self, committed: bool, users: frozenset[str] | None) -> None:
        with self._lock:
            self.pending -= 1
            if committed is True and type(users) is frozenset:
                self._known = users
                self._uncertain = False
            else:
                self._uncertain = True
            self.snapshot = VoicePersonIndexSnapshot(not self.pending and not self._uncertain, self._known)
