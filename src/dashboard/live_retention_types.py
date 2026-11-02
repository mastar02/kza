"""Content-free metadata shared by socket and REST retirement owners."""

from dataclasses import dataclass,field
from enum import StrEnum
from uuid import UUID
from weakref import ref

VOICE_CHANNELS = frozenset({'voices:profiles','voices:captures','voices:review'})


def _control_id(value) -> bool:
    if type(value) is not str or len(value)!=36:
        return False
    try:
        return str(UUID(value))==value
    except ValueError:
        return False


class LiveSubscriptionMode(StrEnum):
    """A server-chosen audience and durable page-clear domain."""

    TECHNICAL = 'technical'
    VOICES_CONTROL_ONLY = 'voices'

    def permits_channel(self, channel: str) -> bool:
        """Technical views retain the existing voice/conversation protocol."""
        return type(channel) is str and (channel in VOICE_CHANNELS or self is LiveSubscriptionMode.TECHNICAL and channel=='conversations')

    def permits_frame(self, frame: dict) -> bool:
        """A voice control socket carries bounded receipt metadata only."""
        if self is LiveSubscriptionMode.TECHNICAL:
            return True
        if type(frame) is not dict or frame.get('ts')!='':
            return False
        kind = frame.get('type')
        if kind=='live_reset' and set(frame)=={'type','payload','ts'}:
            payload = frame['payload']
            return type(payload) is dict and set(payload)=={'id','nonce'} and all(_control_id(value) for value in payload.values())
        if kind=='rest_ready' and set(frame)=={'type','connection_id','channels','ts'}:
            return _control_id(frame['connection_id']) and frame['channels']==sorted(VOICE_CHANNELS)
        if (kind!='rest_retire' or set(frame)!={'type','payload','channel','ts'}
                or type(frame['channel']) is not str or frame['channel'] not in VOICE_CHANNELS):
            return False
        payload = frame['payload']
        if type(payload) is not dict or set(payload)!={'event_ids','receipts'}:
            return False
        tokens, receipts = payload['event_ids'],payload['receipts']
        return (type(tokens) is list and 1<=len(tokens)<=32 and all(_control_id(token) for token in tokens)
            and len(set(tokens))==len(tokens) and type(receipts) is list and len(receipts)<=len(tokens)
            and all(type(row) is dict and set(row)=={'event_id','ticket','nonce'}
                and row['event_id'] in tokens and all(_control_id(value) for value in row.values()) for row in receipts))


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
class RevisionSet:
    """Combine independent content-free producer revision fences."""

    stamps: tuple[RevisionStamp,...]

    def current(self) -> bool:
        """Every producer must still authorize the delivered metadata."""
        return all(stamp.current() for stamp in self.stamps)


@dataclass(frozen=True)
class SnapshotMetadata:
    """A delivered REST batch retains only its token and source lineage."""

    event_id: str
    sources: tuple
    expires_at: float | None = None
    channel: str = 'conversations'
    activity: bool = False
    revision: RevisionStamp | RevisionSet | None = None
    activity_sources: tuple = ()

    def available(self, policy, now: float) -> bool:
        """Raw lifetimes and reviewed metadata use their own source authority."""
        return (not self.activity_sources or policy.activity_sources_available(self.activity_sources)) and (self.expires_at is None or now<self.expires_at) and (
            self.revision is None or self.revision.current()) and (
            policy.activity_sources_available(self.sources) if self.activity else policy.sources_available(self.sources))
