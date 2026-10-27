"""Content-free metadata shared by socket and REST retirement owners."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SnapshotMetadata:
    """A delivered REST batch retains only its token and source lineage."""

    event_id: str
    sources: tuple
