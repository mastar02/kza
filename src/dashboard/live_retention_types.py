"""Content-free metadata shared by socket and REST retirement owners."""

from dataclasses import dataclass,field
from enum import StrEnum
from uuid import UUID
from weakref import ref

VOICE_CHANNELS = frozenset({'voices:profiles','voices:captures','voices:review'})
EVALUATION_CHANNELS = frozenset({'evaluation:sample'})


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
    EVALUATION_CONTROL_ONLY = 'evaluation'

    @property
    def channels(self) -> frozenset[str]:
        if self is LiveSubscriptionMode.EVALUATION_CONTROL_ONLY:
            return EVALUATION_CHANNELS
        return VOICE_CHANNELS | frozenset({'conversations'}) if self is LiveSubscriptionMode.TECHNICAL else VOICE_CHANNELS

    def permits_channel(self, channel: str) -> bool:
        """Keep each server-selected private domain closed."""
        return type(channel) is str and channel in self.channels

    def permits_frame(self, frame: dict) -> bool:
        """Control sockets carry only bounded receipts for their own channels."""
        if self is LiveSubscriptionMode.TECHNICAL:
            return True
        if type(frame) is not dict or frame.get('ts')!='':
            return False
        kind = frame.get('type')
        if kind=='live_reset' and set(frame)=={'type','payload','ts'}:
            payload = frame['payload']
            return type(payload) is dict and set(payload)=={'id','nonce'} and all(_control_id(value) for value in payload.values())
        if kind=='rest_ready' and set(frame)=={'type','connection_id','channels','ts'}:
            return _control_id(frame['connection_id']) and frame['channels']==sorted(self.channels)
        if (kind!='rest_retire' or set(frame)!={'type','payload','channel','ts'}
                or type(frame['channel']) is not str or frame['channel'] not in self.channels):
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
class VoicePersonFence:
    """Only canonical account availability can keep manual person metadata live."""

    authority: ref = field(repr=False)
    identities: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        from src.dashboard.authority import DashboardAuthority

        if (type(self.authority) is not ref or type(self.authority()) is not DashboardAuthority
                or type(self.identities) is not tuple or not 1 <= len(self.identities) <= 50
                or any(type(identity) is not str or not 1 <= len(identity) <= 128
                       or any(ord(char) < 32 or ord(char) == 127 for char in identity)
                       for identity in self.identities)
                or tuple(sorted(set(self.identities))) != self.identities):
            raise ValueError('invalid_voice_person_fence')

    def current(self) -> bool:
        """Pending retirement, unknown membership or a lost owner retires the view."""
        owner = self.authority()
        try:
            return owner is not None and owner.users_linkable(frozenset(self.identities)) is True
        except Exception:
            return False


@dataclass(frozen=True)
class EvaluationScopeFence:
    """Keep the original grant, actual typed sources, proof hashes and deadline."""

    actor: object = field(repr=False)
    source_digests: frozenset[str] = field(repr=False)
    expires_at: float
    sources: tuple = field(repr=False)

    def __post_init__(self) -> None:
        import math
        from src.dashboard.evaluation_actor import EvaluationReadActor, evaluation_digests
        from src.memory.fact_ledger import MemorySource
        if (type(self.actor) is not EvaluationReadActor or type(self.expires_at) not in (int,float)
                or not math.isfinite(self.expires_at) or type(self.sources) is not tuple
                or not 1<=len(self.sources)<=256 or any(type(source) is not MemorySource for source in self.sources)
                or sum(len(source.fact_dependencies) for source in self.sources)>256
                or self.expires_at > min(self.actor.grant_deadline(),*[source.expires_at for source in self.sources])):
            raise ValueError('invalid_evaluation_fence')
        evaluation_digests(self.source_digests)

    def current(self) -> bool:
        return self.actor.current(self.source_digests,self.expires_at)

@dataclass(frozen=True)
class RevisionSet:
    """Combine closed, content-free revision and canonical availability fences."""

    stamps: tuple[RevisionStamp | VoicePersonFence | EvaluationScopeFence, ...]

    def __post_init__(self) -> None:
        if type(self.stamps) is not tuple or any(type(stamp) not in (RevisionStamp, VoicePersonFence, EvaluationScopeFence) for stamp in self.stamps):
            raise ValueError('invalid_revision_set')

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
    empty_metadata: bool = False

    def available(self, policy, now: float) -> bool:
        """Raw lifetimes and reviewed metadata use their own source authority."""
        empty_page = (self.empty_metadata is True and self.sources == () and not self.activity_sources
            and not self.activity and self.channel in {'voices:profiles','voices:captures'}
            and type(self.revision) in (RevisionStamp,RevisionSet))
        return (not self.activity_sources or policy.activity_sources_available(self.activity_sources)) and (self.expires_at is None or now<self.expires_at) and (
            self.revision is None or self.revision.current()) and (
            policy.activity_sources_available(self.sources) if self.activity or empty_page else policy.sources_available(self.sources))
