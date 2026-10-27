"""Content-free metadata shared by socket and REST retirement owners."""

from dataclasses import dataclass,field
from weakref import ref


@dataclass(frozen=True)
class RevisionStamp:
    """A weak content-free invalidation fence for a producer's metadata views."""

    producer: ref = field(repr=False)
    revision: int

    def current(self) -> bool:
        """A lost producer or any intervening mutation retires the old view."""
        owner = self.producer()
        return owner is not None and owner.projection_revision==self.revision


@dataclass(frozen=True)
class SnapshotMetadata:
    """A delivered REST batch retains only its token and source lineage."""

    event_id: str
    sources: tuple
    expires_at: float | None = None
    channel: str = 'conversations'
    activity: bool = False
    revision: RevisionStamp | None = None
    activity_sources: tuple = ()

    def available(self, policy, now: float) -> bool:
        """Raw lifetimes and reviewed metadata use their own source authority."""
        return (not self.activity_sources or policy.activity_sources_available(self.activity_sources)) and (self.expires_at is None or now<self.expires_at) and (
            self.revision is None or self.revision.current()) and (
            policy.activity_sources_available(self.sources) if self.activity else policy.sources_available(self.sources))
