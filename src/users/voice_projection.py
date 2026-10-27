"""Detached private voice snapshots; no model vectors cross the HTTP boundary."""

from dataclasses import dataclass,field

from src.dashboard.live_retention_types import RevisionSet,RevisionStamp


@dataclass(repr=False)
class VoiceProjection:
    """A native reader transfers its bounded payload to the actual send owner."""

    rows: list = field(repr=False)
    sources: tuple
    observed_at: float
    expires_at: float
    channel: str
    next_cursor: str | None = None
    profile_refs: tuple = ()
    revision: RevisionStamp | RevisionSet | None = None
    associated_profiles: tuple = ()
    activity_sources: tuple = ()
