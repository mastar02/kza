"""Immutable admission and retirement contract for new consented recordings.

No recording, training, file access or runtime retention setting is changed by
this module. Private registry and erasure adapters must apply these decisions
before exposing bytes, and retain failed cleanup entries until deletion succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import math
from pathlib import PurePosixPath
import re

from src.evaluation.provenance import CaptureProvenance, FrameInterval


class CorpusSplit(str, Enum):
    """A complete recording session's immutable evaluation partition."""

    CALIBRATION = "calibration"
    RESERVED = "reserved"


def _aware(value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone_required")


def _instant(value: datetime) -> datetime:
    """Return the UTC instant, including disambiguation of folded local time."""
    _aware(value)
    return value.astimezone(timezone.utc)


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
        if not _instant(self.consented_at) <= _instant(self.started_at) < _instant(self.ended_at):
            raise ValueError("invalid_session_interval")
        if self.revoked_at is not None:
            _aware(self.revoked_at)


@dataclass(frozen=True)
class CorpusCapture:
    """Original recording metadata; ordinary captures cannot be imported.

    channel and dsp_revision are legacy display labels. Only the typed
    provenance participates in comparison admission; labels cannot supply it.
    """

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
    provenance: CaptureProvenance | None = None

    def __post_init__(self) -> None:
        _required(self.capture_id, self.session_id, self.channel, self.dsp_revision,
                  self.room, self.condition, self.candidate_revision)
        _aware(self.captured_at)
        _aware(self.original_expires_at)
        _digest(self.recording_sha256)
        if self.origin != "new_consented_evaluation":
            raise ValueError("ordinary_capture_not_admitted")
        captured_at = _instant(self.captured_at)
        if not captured_at < _instant(self.original_expires_at) <= captured_at + timedelta(days=30):
            raise ValueError("original_deadline_exceeds_consent")
        if type(self.duration_ms) is not int or self.duration_ms <= 0:
            raise ValueError("invalid_duration")
        if type(self.distance_m) not in {int, float} or not math.isfinite(self.distance_m) or self.distance_m <= 0:
            raise ValueError("invalid_distance")
        if self.provenance is not None:
            if type(self.provenance) is not CaptureProvenance:
                raise ValueError("invalid_capture_provenance")
            CaptureProvenance.__post_init__(self.provenance)
            _, duration_ms = FrameInterval(0, self.provenance.frame_count).millisecond_bounds(
                self.provenance.sample_rate_hz)
            if self.duration_ms != duration_ms:
                raise ValueError("capture_duration_does_not_match_pcm")


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
    audio_artifact_id: str | None = None
    frame_interval: FrameInterval | None = None

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
        if self.audio_artifact_id is not None:
            _required(self.audio_artifact_id)
        if self.frame_interval is not None:
            if type(self.frame_interval) is not FrameInterval:
                raise ValueError("invalid_artifact_frames")
            FrameInterval.__post_init__(self.frame_interval)


@dataclass(frozen=True)
class CorpusComparison:
    """Explicit pairing, never a request to crop or synthesize reference text."""

    comparison_id: str
    audio_artifact_id: str
    reference_artifact_id: str
    hypothesis_artifact_id: str

    def __post_init__(self) -> None:
        _required(self.comparison_id, self.audio_artifact_id,
                  self.reference_artifact_id, self.hypothesis_artifact_id)
        if len({self.audio_artifact_id, self.reference_artifact_id, self.hypothesis_artifact_id}) != 3:
            raise ValueError("comparison_artifacts_must_be_distinct")


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
    schema_version: int = 1
    comparisons: tuple[CorpusComparison, ...] = ()

    def __post_init__(self) -> None:
        _required(self.protocol_revision)
        reviewed_at = _instant(self.reviewed_at)
        if self.purpose != "consented_calibration_and_evaluation":
            raise ValueError("invalid_corpus_purpose")
        if type(self.schema_version) is not int or self.schema_version not in (1, 2):
            raise ValueError("unsupported_corpus_schema")
        for values, record_type in ((self.sessions, CorpusSession), (self.captures, CorpusCapture), (self.artifacts, CorpusArtifact)):
            if type(values) is not tuple:
                raise ValueError("immutable_records_required")
            if any(type(item) is not record_type for item in values):
                raise ValueError("invalid_corpus_record")
        if (type(self.comparisons) is not tuple
                or any(type(item) is not CorpusComparison for item in self.comparisons)):
            raise ValueError("invalid_corpus_comparisons")
        if self.schema_version == 1 and (self.comparisons
                or any(item.provenance is not None for item in self.captures)
                or any(item.audio_artifact_id is not None or item.frame_interval is not None for item in self.artifacts)):
            raise ValueError("schema_one_has_no_typed_alignment")
        sessions = {item.session_id: item for item in self.sessions}
        captures = {item.capture_id: item for item in self.captures}
        artifacts = {item.artifact_id: item for item in self.artifacts}
        comparisons = {item.comparison_id: item for item in self.comparisons}
        if len(sessions) != len(self.sessions) or len(captures) != len(self.captures) or len(artifacts) != len(self.artifacts):
            raise ValueError("duplicate_record")
        if len(comparisons) != len(self.comparisons):
            raise ValueError("duplicate_comparison")
        if len({item.recording_sha256 for item in self.captures}) != len(self.captures):
            raise ValueError("source_reimported")
        if len({item.relative_path for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("duplicate_artifact_path")
        if self.closed_at is not None:
            _aware(self.closed_at)
            if _instant(self.closed_at) > reviewed_at:
                raise ValueError("future_closure")
        for session in self.sessions:
            CorpusSession.__post_init__(session)
            if _instant(session.ended_at) > reviewed_at or (session.revoked_at is not None and _instant(session.revoked_at) > reviewed_at):
                raise ValueError("future_session_event")
        for capture in self.captures:
            CorpusCapture.__post_init__(capture)
            session = sessions.get(capture.session_id)
            if session is None or not _instant(session.started_at) <= _instant(capture.captured_at) < _instant(session.ended_at):
                raise ValueError("capture_outside_session")
            if capture.provenance is None:
                outside_session = _instant(capture.captured_at) + timedelta(milliseconds=capture.duration_ms) > _instant(session.ended_at)
            else:
                elapsed = session.ended_at.astimezone(timezone.utc) - capture.captured_at.astimezone(timezone.utc)
                microseconds = (elapsed.days * 86400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds
                outside_session = capture.provenance.frame_count * 1_000_000 > microseconds * capture.provenance.sample_rate_hz
            if outside_session:
                raise ValueError("recording_outside_session")
            originals = [item for item in self.artifacts
                         if item.capture_id == capture.capture_id and item.kind == "recording"]
            if (len(originals) != 1 or originals[0].parent_id is not None
                    or originals[0].sha256 != capture.recording_sha256
                    or (originals[0].start_ms, originals[0].end_ms) != (0, capture.duration_ms)):
                raise ValueError("original_recording_not_registered")
            if capture.provenance is not None and originals[0].frame_interval != FrameInterval(0, capture.provenance.frame_count):
                raise ValueError("original_pcm_frames_not_registered")
        for artifact in self.artifacts:
            CorpusArtifact.__post_init__(artifact)
            capture = captures.get(artifact.capture_id)
            if capture is None or artifact.end_ms > capture.duration_ms:
                raise ValueError("artifact_outside_capture")
            if artifact.kind != "recording" and artifact.parent_id is None:
                raise ValueError("derivative_missing_parent")
            if artifact.frame_interval is not None:
                if capture.provenance is None:
                    raise ValueError("artifact_frames_need_capture_provenance")
                if artifact.frame_interval.end_frame > capture.provenance.frame_count:
                    raise ValueError("artifact_frames_outside_capture")
                if artifact.frame_interval.millisecond_bounds(capture.provenance.sample_rate_hz) != (artifact.start_ms, artifact.end_ms):
                    raise ValueError("artifact_milliseconds_do_not_match_pcm")
            if artifact.audio_artifact_id is not None:
                audio = artifacts.get(artifact.audio_artifact_id)
                if (artifact.kind not in {"human_reference", "stt_hypothesis"}
                        or audio is None or audio.kind not in {"recording", "clip"}
                        or artifact.parent_id != audio.artifact_id or artifact.capture_id != audio.capture_id):
                    raise ValueError("invalid_explicit_audio_binding")
            seen = {artifact.artifact_id}
            child = artifact
            while child.parent_id is not None:
                parent = artifacts.get(child.parent_id)
                if parent is None or parent.artifact_id in seen:
                    raise ValueError("missing_or_cyclic_parent")
                if parent.capture_id != child.capture_id or not parent.start_ms <= child.start_ms < child.end_ms <= parent.end_ms:
                    raise ValueError("derivative_changes_source")
                if child.frame_interval is not None:
                    if parent.frame_interval is None:
                        raise ValueError("derivative_pcm_parent_unknown")
                    if not parent.frame_interval.start_frame <= child.frame_interval.start_frame < child.frame_interval.end_frame <= parent.frame_interval.end_frame:
                        raise ValueError("derivative_changes_pcm_interval")
                seen.add(parent.artifact_id)
                child = parent
        for comparison in self.comparisons:
            CorpusComparison.__post_init__(comparison)
            self._comparison_artifacts(comparison)

    def _comparison_artifacts(self, comparison: CorpusComparison) -> tuple[CorpusArtifact, CorpusArtifact, CorpusArtifact]:
        artifacts = {item.artifact_id: item for item in self.artifacts}
        try:
            audio, reference, hypothesis = (artifacts[identity] for identity in (
                comparison.audio_artifact_id, comparison.reference_artifact_id,
                comparison.hypothesis_artifact_id))
        except KeyError:
            raise ValueError("comparison_artifact_missing") from None
        if (audio.kind not in {"recording", "clip"} or reference.kind != "human_reference"
                or hypothesis.kind != "stt_hypothesis"):
            raise ValueError("invalid_comparison_kind")
        if len({audio.capture_id, reference.capture_id, hypothesis.capture_id}) != 1:
            raise ValueError("comparison_crosses_capture")
        capture = next(item for item in self.captures if item.capture_id == audio.capture_id)
        if capture.provenance is None or not capture.provenance.comparable:
            raise ValueError("comparison_provenance_not_observed")
        if (audio.frame_interval is None
                or reference.frame_interval != audio.frame_interval
                or hypothesis.frame_interval != audio.frame_interval
                or (reference.start_ms, reference.end_ms) != (audio.start_ms, audio.end_ms)
                or (hypothesis.start_ms, hypothesis.end_ms) != (audio.start_ms, audio.end_ms)):
            raise ValueError("comparison_interval_mismatch")
        if any(item.audio_artifact_id != audio.artifact_id or item.parent_id != audio.artifact_id
               for item in (reference, hypothesis)):
            raise ValueError("comparison_audio_binding_mismatch")
        return audio, reference, hypothesis

    def require_comparison(self, comparison_id: str, now: datetime) -> CorpusComparison:
        """Validate a live comparison in this immutable snapshot, without I/O.

        This is NOT current runtime authorization. Before opening any private
        artifact an adapter must revalidate durable consent/tombstones and the
        actual file identity; a restored snapshot cannot establish that state.
        No reference or hypothesis is cropped or derived by this method.
        """
        self.__post_init__()
        instant = _instant(now)
        if instant < _instant(self.reviewed_at):
            raise ValueError("clock_before_review")
        comparison = next((item for item in self.comparisons if item.comparison_id == comparison_id), None)
        if comparison is None:
            raise ValueError("unknown_comparison")
        audio, _, _ = self._comparison_artifacts(comparison)
        if instant >= self.deadline(audio.capture_id):
            raise PermissionError("comparison_retired")
        return comparison

    def deadline(self, capture_id: str) -> datetime:
        """Return the earliest original, consent or evaluation instant in UTC."""
        capture = next(item for item in self.captures if item.capture_id == capture_id)
        session = next(item for item in self.sessions if item.session_id == capture.session_id)
        limits = [_instant(capture.original_expires_at), _instant(capture.captured_at) + timedelta(days=30)]
        limits.extend(_instant(item.revoked_at) for item in self.sessions
                      if item.participant_ref == session.participant_ref and item.revoked_at is not None)
        if self.closed_at is not None:
            limits.append(_instant(self.closed_at))
        return min(limits)

    def retired_artifacts(self, now: datetime) -> tuple[CorpusArtifact, ...]:
        """Plan all expired bytes, including references/copies/embeddings.

        Readers deny these immediately. A purge adapter must retain each entry
        until deletion is confirmed; this method does not claim deletion.
        """
        instant = _instant(now)
        if instant < _instant(self.reviewed_at):
            raise ValueError("clock_before_review")
        return tuple(item for item in self.artifacts if instant >= self.deadline(item.capture_id))
