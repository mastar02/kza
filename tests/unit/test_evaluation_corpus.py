"""Offline fixtures exercise the independent evaluation corpus policy."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.evaluation.corpus import (
    CorpusArtifact,
    CorpusCapture,
    CorpusManifest,
    CorpusSession,
    CorpusSplit,
)


CAPTURED = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)


@pytest.fixture
def manifest():
    session = CorpusSession(
        "session-a", "participant-a", "consent-a", CAPTURED - timedelta(minutes=1),
        CAPTURED, CAPTURED + timedelta(minutes=10), CorpusSplit.RESERVED,
    )
    capture = CorpusCapture(
        "capture-a", session.session_id, CAPTURED, CAPTURED + timedelta(days=30),
        "a" * 64, 30_000, "input-1/channel-1", "dsp-revision-a", "Living", 3,
        "clear", "candidate-a",
    )
    original = CorpusArtifact("audio-a", capture.capture_id, "audio/a.wav", "a" * 64,
                              "recording", 0, 30_000)
    clip = CorpusArtifact("clip-a", capture.capture_id, "audio/clip.wav", "b" * 64,
                          "clip", 1000, 7000, original.artifact_id)
    reference = CorpusArtifact("reference-a", capture.capture_id, "references/a.json", "c" * 64,
                               "human_reference", 1000, 7000, clip.artifact_id, "review-a")
    return CorpusManifest("protocol-a", CAPTURED + timedelta(hours=1),
                          (session,), (capture,), (original, clip, reference))


def test_copy_clip_reference_and_embedding_keep_original_deadline(manifest):
    copy = replace(manifest.artifacts[1], artifact_id="copy-a", relative_path="copies/a.wav")
    embedding = replace(copy, artifact_id="embedding-a", relative_path="embeddings/a.bin", kind="embedding")
    revised = replace(manifest, reviewed_at=CAPTURED + timedelta(days=29),
                      artifacts=(*manifest.artifacts, copy, embedding))
    assert revised.retired_artifacts(CAPTURED + timedelta(days=30) - timedelta(microseconds=1)) == ()
    assert revised.retired_artifacts(CAPTURED + timedelta(days=30)) == revised.artifacts


@pytest.mark.parametrize("event", ["revoke", "close"])
def test_consent_revocation_and_closure_retire_all_artifacts_immediately(manifest, event):
    stopped = CAPTURED + timedelta(minutes=30)
    if event == "revoke":
        revised = replace(manifest, sessions=(replace(manifest.sessions[0], revoked_at=stopped),))
    else:
        revised = replace(manifest, closed_at=stopped)
    assert revised.deadline("capture-a") == stopped
    assert revised.retired_artifacts(revised.reviewed_at) == revised.artifacts


def test_shorter_original_expiry_is_preserved(manifest):
    expires = CAPTURED + timedelta(days=2)
    revised = replace(manifest, captures=(replace(manifest.captures[0], original_expires_at=expires),))
    assert revised.deadline("capture-a") == expires


def test_one_session_cannot_be_assigned_to_both_partitions(manifest):
    conflicting = replace(manifest.sessions[0], split=CorpusSplit.CALIBRATION)
    with pytest.raises(ValueError, match="duplicate_record"):
        replace(manifest, sessions=(*manifest.sessions, conflicting))


def test_same_recording_cannot_be_reimported_into_calibration(manifest):
    other_session = replace(manifest.sessions[0], session_id="session-b", split=CorpusSplit.CALIBRATION)
    replay = replace(manifest.captures[0], capture_id="capture-b", session_id="session-b")
    with pytest.raises(ValueError, match="source_reimported"):
        replace(manifest, sessions=(*manifest.sessions, other_session), captures=(*manifest.captures, replay))


def test_derivative_cannot_cross_to_another_source(manifest):
    other = replace(manifest.captures[0], capture_id="capture-b", recording_sha256="d" * 64)
    other_original = replace(manifest.artifacts[0], artifact_id="audio-b", capture_id="capture-b",
                             relative_path="audio/b.wav", sha256="d" * 64)
    changed = replace(manifest.artifacts[1], capture_id="capture-b")
    with pytest.raises(ValueError, match="derivative_changes_source"):
        replace(manifest, captures=(*manifest.captures, other),
                artifacts=(manifest.artifacts[0], other_original, changed))


@pytest.mark.parametrize("origin", ["ordinary_capture", "legacy", "training"])
def test_ordinary_data_is_not_renewed_as_evaluation(manifest, origin):
    with pytest.raises(ValueError, match="ordinary_capture_not_admitted"):
        replace(manifest.captures[0], origin=origin)


def test_capture_expiry_cannot_exceed_thirty_days(manifest):
    with pytest.raises(ValueError, match="original_deadline_exceeds_consent"):
        replace(manifest.captures[0], original_expires_at=CAPTURED + timedelta(days=31))


@pytest.mark.parametrize("path", ["../outside.wav", "/outside.wav", "audio/../../outside.wav",
                                  "..\\outside.wav", ".", "./audio/a.wav", "audio//a.wav", "a\x00.wav"])
def test_artifact_paths_cannot_escape_private_root(manifest, path):
    with pytest.raises(ValueError, match="path_outside_corpus"):
        replace(manifest.artifacts[0], relative_path=path)


def test_missing_parent_and_cycles_rejected(manifest):
    for parent in ("missing", "clip-a"):
        changed = replace(manifest.artifacts[1], parent_id=parent)
        with pytest.raises(ValueError, match="missing_or_cyclic_parent"):
            replace(manifest, artifacts=(manifest.artifacts[0], changed))


def test_clip_and_human_reference_must_stay_in_source_interval(manifest):
    changed = replace(manifest.artifacts[2], end_ms=30_000)
    with pytest.raises(ValueError, match="derivative_changes_source"):
        replace(manifest, artifacts=(*manifest.artifacts[:2], changed))
    with pytest.raises(ValueError, match="missing_provenance"):
        replace(manifest.artifacts[2], human_review_ref=None)


def test_future_capture_session_and_naive_clock_rejected(manifest):
    with pytest.raises(ValueError, match="future_session_event"):
        replace(manifest, reviewed_at=CAPTURED)
    with pytest.raises(ValueError, match="timezone_required"):
        manifest.retired_artifacts(datetime(2026, 12, 3))
    with pytest.raises(ValueError, match="clock_before_review"):
        manifest.retired_artifacts(CAPTURED)
    with pytest.raises(ValueError, match="capture_outside_session"):
        replace(manifest, captures=(replace(manifest.captures[0], captured_at=CAPTURED + timedelta(hours=1)),))


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -1, True])
def test_invalid_acoustic_conditions_rejected(manifest, distance):
    with pytest.raises(ValueError, match="invalid_distance"):
        replace(manifest.captures[0], distance_m=distance)


def test_revocation_covers_every_session_of_participant(manifest):
    revoked_at = CAPTURED + timedelta(minutes=30)
    session_a = replace(manifest.sessions[0], revoked_at=revoked_at)
    session_b = replace(manifest.sessions[0], session_id="session-b")
    capture_b = replace(manifest.captures[0], capture_id="capture-b", session_id="session-b",
                        recording_sha256="d" * 64)
    original_b = replace(manifest.artifacts[0], artifact_id="audio-b", capture_id="capture-b",
                         relative_path="audio/b.wav", sha256="d" * 64)
    revised = replace(manifest, sessions=(session_a, session_b),
                      captures=(*manifest.captures, capture_b), artifacts=(*manifest.artifacts, original_b))
    assert revised.deadline("capture-b") == revoked_at
    assert revised.retired_artifacts(revised.reviewed_at) == revised.artifacts


@pytest.mark.parametrize("change", ["omitted", "hash", "interval", "parent"])
def test_original_recording_must_be_registered_consistently(manifest, change):
    original = manifest.artifacts[0]
    if change == "omitted":
        artifacts = manifest.artifacts[1:]
    else:
        kwargs = {"hash": {"sha256": "d" * 64}, "interval": {"end_ms": 29_000},
                  "parent": {"parent_id": "clip-a"}}[change]
        artifacts = (replace(original, **kwargs), *manifest.artifacts[1:])
    with pytest.raises(ValueError, match="original_recording_not_registered"):
        replace(manifest, artifacts=artifacts)


def test_derivative_cannot_be_registered_without_lineage(manifest):
    with pytest.raises(ValueError, match="derivative_missing_parent"):
        replace(manifest, artifacts=(manifest.artifacts[0], replace(manifest.artifacts[1], parent_id=None)))
