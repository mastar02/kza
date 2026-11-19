"""Detached private voice snapshots; no model vectors cross the HTTP boundary."""

from dataclasses import dataclass,field

from src.dashboard.live_retention_types import RevisionSet,RevisionStamp


@dataclass(frozen=True)
class VoiceQuerySummary:
    """Content-free envelope of a filtered captures page (T044-U1).

    ``matching_count`` is the number of matching live rows in the whole filtered
    window, known only when ``scan_complete``; ``room_options`` are room ids of
    live rows matching every filter except the room, only for filters-only
    requests (``None`` for a search). Each value is hidden at its own deadline.
    """

    matching_count: int | None
    scan_complete: bool
    count_expires_at: float | None
    room_options: tuple[str, ...] | None
    room_options_expires_at: float | None


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
    summary: VoiceQuerySummary | None = None
