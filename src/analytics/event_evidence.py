"""Content-free receipts for activity metadata used by pattern analysis.

These receipts are references, not authorization to publish or persist a
suggestion. Consumers must resolve them against the actual policy event store.
Raw capture expiry does not extend to numeric activity metadata; conversation
revocation and semantic fact dependencies still apply to that metadata.
"""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import re

from src.analytics.event_storage import event_sources


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class EventDependency:
    """Bind a derived pattern to one row at its original metadata deadline."""

    target_id: str
    event_id: int
    revision: str
    expires_at: float

    def __post_init__(self) -> None:
        for value in (self.target_id, self.revision):
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("invalid_event_dependency")
        if type(self.event_id) is not int or not 0 < self.event_id <= 2**63 - 1:
            raise ValueError("invalid_event_dependency")
        if type(self.expires_at) not in (int, float) or not math.isfinite(self.expires_at):
            raise ValueError("invalid_event_dependency")

    def to_dict(self) -> dict:
        """Serialize without copying source IDs, paths, names or transcripts."""
        return {
            "target_id": self.target_id,
            "event_id": self.event_id,
            "revision": self.revision,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "EventDependency":
        """Reject unknown receipt protocols instead of guessing provenance."""
        if type(value) is not dict or set(value) != {
            "target_id", "event_id", "revision", "expires_at"
        }:
            raise ValueError("invalid_event_dependency")
        return cls(**value)


@dataclass(frozen=True)
class PatternEvent:
    """Immutable activity projection; free text never enters pattern analysis."""

    timestamp: float
    entity_id: str
    action: str
    dependency: EventDependency

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)

    @property
    def hour(self) -> int:
        return self.datetime.hour

    @property
    def minute(self) -> int:
        return self.datetime.minute

    @property
    def is_weekend(self) -> bool:
        return self.datetime.weekday() >= 5


@dataclass(frozen=True)
class PatternHistory:
    """One bounded SQL observation, including whether its history is complete."""

    events: tuple[PatternEvent, ...]
    complete: bool


def event_target_id(descriptor: dict) -> str:
    """Identify the canonical store without exposing its private location."""
    return _digest(descriptor)


def project_pattern_event(row, target_id: str) -> PatternEvent:
    """Fingerprint metadata and lineage, independently of raw text scrubbing."""
    sources = event_sources(row["sources_json"], False)
    # Changes to a semantic dependency must invalidate activity derivatives even
    # when their own configured metadata lifetime is longer than that fact's.
    deadline = min(
        [float(row["expires_at"])]
        + [d.expires_at for s in sources for d in s.fact_dependencies if d.expires_at is not None]
    )
    metadata = {
        key: row[key] for key in (
            "timestamp", "event_type", "entity_id", "action", "user_id", "expires_at"
        )
    }
    metadata["sources"] = [source.to_dict() for source in sources]
    return PatternEvent(
        float(row["timestamp"]), row["entity_id"], row["action"],
        EventDependency(target_id, row["id"], _digest(metadata), deadline),
    )
