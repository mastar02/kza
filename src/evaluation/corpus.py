"""Immutable admission and retirement contract for new consented recordings.

No recording, training, file access or runtime retention setting is changed by
this module. Private registry and erasure adapters must apply these decisions
before exposing bytes, and retain failed cleanup entries until deletion succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import math
from pathlib import PurePosixPath
import re


class CorpusSplit(str, Enum):
    """A complete recording session's immutable evaluation partition."""

    CALIBRATION = "calibration"
    RESERVED = "reserved"


def _aware(value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone_required")


def _required(*values: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("missing_provenance")


def _digest(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("invalid_sha256")


@dataclass(frozen=True)
class CorpusSession:
    """Consent and partition of a complete session, using private references."""

    session_id: str
    participant_ref: str
    consent_ref: str
    consented_at: datetime
    started_at: datetime
    ended_at: datetime
    split: CorpusSplit
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        _required(self.session_id, self.participant_ref, self.consent_ref)
        for value in (self.consented_at, self.started_at, self.ended_at):
            _aware(value)
        if not isinstance(self.split, CorpusSplit):
            raise ValueError("invalid_split")
        if not self.consented_at <= self.started_at < self.ended_at:
            raise ValueError("invalid_session_interval")
        if self.revoked_at is not None:
            _aware(self.revoked_at)


@dataclass(frozen=True)
class CorpusCapture:
    """Original recording provenance; ordinary captures cannot be imported."""

    capture_id: str
    session_id: str
    captured_at: datetime
    original_expires_at: datetime
    recording_sha256: str
    duration_ms: int
    channel: str
    dsp_revision: str
    room: str
    distance_m: float
    condition: str
    candidate_revision: str
    origin: str = "new_consented_evaluation"

    def __post_init__(self) -> None:
        _required(self.capture_id, self.session_id, self.channel, self.dsp_revision,
                  self.room, self.condition, self.candidate_revision)
        _aware(self.captured_at)
        _aware(self.original_expires_at)
        _digest(self.recording_sha256)
        if self.origin != "new_consented_evaluation":
            raise ValueError("ordinary_capture_not_admitted")
        if not self.captured_at < self.original_expires_at <= self.captured_at + timedelta(days=30):
            raise ValueError("original_deadline_exceeds_consent")
        if type(self.duration_ms) is not int or self.duration_ms <= 0:
            raise ValueError("invalid_duration")
        if type(self.distance_m) not in {int, float} or not math.isfinite(self.distance_m) or self.distance_m <= 0:
            raise ValueError("invalid_distance")


@dataclass(frozen=True)
class CorpusArtifact:
    """Registered bytes, copy, reference or derivative of one source interval.

    No split, capture time or new deadline may be supplied on a derivative.
    Paths are relative to the private corpus root; adapters must also reject
    symlink traversal when opening or deleting files.
    """

    artifact_id: str
    capture_id: str
    relative_path: str
    sha256: str
    kind: str
    start_ms: int
    end_ms: int
    parent_id: str | None = None
    human_review_ref: str | None = None

    def __post_init__(self) -> None:
        _required(self.artifact_id, self.capture_id, self.relative_path, self.kind)
        _digest(self.sha256)
        path = PurePosixPath(self.relative_path)
        if (path.is_absolute() or ".." in path.parts or "\\" in self.relative_path
                or "\x00" in self.relative_path or path == PurePosixPath(".")
                or str(path) != self.relative_path):
            raise ValueError("path_outside_corpus")
        if type(self.start_ms) is not int or type(self.end_ms) is not int or not 0 <= self.start_ms < self.end_ms:
            raise ValueError("invalid_source_interval")
        if self.kind == "human_reference":
            _required(self.human_review_ref)


@dataclass(frozen=True)
class CorpusManifest:
    """Validated private snapshot for admission, availability and purge planning.

    Registry writers must preserve session IDs/partitions and original captures
    across snapshots. Closure/revocation is monotonic and must survive restore;
    validating a stale snapshot alone does not establish current consent.
    """

    protocol_revision: str
    reviewed_at: datetime
    sessions: tuple[CorpusSession, ...]
    captures: tuple[CorpusCapture, ...]
    artifacts: tuple[CorpusArtifact, ...]
    closed_at: datetime | None = None
    purpose: str = "consented_calibration_and_evaluation"

    def __post_init__(self) -> None:
        _required(self.protocol_revision)
        _aware(self.reviewed_at)
        if self.purpose != "consented_calibration_and_evaluation":
            raise ValueError("invalid_corpus_purpose")
        for values in (self.sessions, self.captures, self.artifacts):
            if type(values) is not tuple:
                raise ValueError("immutable_records_required")
        sessions = {item.session_id: item for item in self.sessions}
        captures = {item.capture_id: item for item in self.captures}
        artifacts = {item.artifact_id: item for item in self.artifacts}
        if len(sessions) != len(self.sessions) or len(captures) != len(self.captures) or len(artifacts) != len(self.artifacts):
            raise ValueError("duplicate_record")
        if len({item.recording_sha256 for item in self.captures}) != len(self.captures):
            raise ValueError("source_reimported")
        if len({item.relative_path for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("duplicate_artifact_path")
        if self.closed_at is not None:
            _aware(self.closed_at)
            if self.closed_at > self.reviewed_at:
                raise ValueError("future_closure")
        for session in self.sessions:
            if session.ended_at > self.reviewed_at or (session.revoked_at is not None and session.revoked_at > self.reviewed_at):
                raise ValueError("future_session_event")
        for capture in self.captures:
            session = sessions.get(capture.session_id)
            if session is None or not session.started_at <= capture.captured_at < session.ended_at:
                raise ValueError("capture_outside_session")
            if capture.captured_at + timedelta(milliseconds=capture.duration_ms) > session.ended_at:
                raise ValueError("recording_outside_session")
            originals = [item for item in self.artifacts
                         if item.capture_id == capture.capture_id and item.kind == "recording"]
            if (len(originals) != 1 or originals[0].parent_id is not None
                    or originals[0].sha256 != capture.recording_sha256
                    or (originals[0].start_ms, originals[0].end_ms) != (0, capture.duration_ms)):
                raise ValueError("original_recording_not_registered")
        for artifact in self.artifacts:
            capture = captures.get(artifact.capture_id)
            if capture is None or artifact.end_ms > capture.duration_ms:
                raise ValueError("artifact_outside_capture")
            if artifact.kind != "recording" and artifact.parent_id is None:
                raise ValueError("derivative_missing_parent")
            seen = {artifact.artifact_id}
            child = artifact
            while child.parent_id is not None:
                parent = artifacts.get(child.parent_id)
                if parent is None or parent.artifact_id in seen:
                    raise ValueError("missing_or_cyclic_parent")
                if parent.capture_id != child.capture_id or not parent.start_ms <= child.start_ms < child.end_ms <= parent.end_ms:
                    raise ValueError("derivative_changes_source")
                seen.add(parent.artifact_id)
                child = parent

    def deadline(self, capture_id: str) -> datetime:
        """Return the earliest original, consent or evaluation deadline."""
        capture = next(item for item in self.captures if item.capture_id == capture_id)
        session = next(item for item in self.sessions if item.session_id == capture.session_id)
        limits = [capture.original_expires_at, capture.captured_at + timedelta(days=30)]
        limits.extend(item.revoked_at for item in self.sessions
                      if item.participant_ref == session.participant_ref and item.revoked_at is not None)
        if self.closed_at is not None:
            limits.append(self.closed_at)
        return min(limits)

    def retired_artifacts(self, now: datetime) -> tuple[CorpusArtifact, ...]:
        """Plan all expired bytes, including references/copies/embeddings.

        Readers deny these immediately. A purge adapter must retain each entry
        until deletion is confirmed; this method does not claim deletion.
        """
        _aware(now)
        if now < self.reviewed_at:
            raise ValueError("clock_before_review")
        return tuple(item for item in self.artifacts if now >= self.deadline(item.capture_id))
