"""Offline fixtures exercise the independent evaluation corpus policy."""

import ast
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone, tzinfo
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.task_outcome import EffectState, TaskOutcome, TaskStatus
from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult
from src.evaluation.corpus import (
    CorpusArtifact,
    CorpusCapture,
    CorpusComparison,
    CorpusManifest,
    CorpusSession,
    CorpusSplit,
)
from src.evaluation.freeze_manifest import (
    ConsentReceipt, EvaluationClip, Partition, VoiceAssignment, WakeCounts,
    canonical_sha256, clarity_median, complete_delivery, complete_task, nearest_rank,
    recognition, seeded_order, validate_inventory, voice_pair_counts,
)
from src.evaluation.provenance import (
    CaptureLane, CaptureProvenance, CaptureSource, DSPCertainty, DSPObservation,
    FrameInterval, MAX_COUNTER,
)
from src.nlu.household_intents import HouseholdFamily, HouseholdIntentClassifier, HouseholdOperation
from src.pipeline.wake_verification import WakeCheckCode, WakeVerificationConfig, WakeVerificationMode, WakeVerifier
from tools.ambient_wer import build_report
from tools.analyze_audio_quality import measure_channel


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


def observed_provenance(**changes):
    values = dict(source=CaptureSource.LIVE_INPUT, source_ref="input-a",
                  lane=CaptureLane.COMMAND, stream_generation=7,
                  segment_generation=23, source_channels=2, selected_channel=1,
                  sample_rate_hz=16_000, frame_count=480_000,
                  dsp=DSPObservation(DSPCertainty.OBSERVED, "declared-a", "observed-a"))
    values.update(changes)
    return CaptureProvenance(**values)


def aligned_manifest(manifest, *, full=False, dsp=None, comparisons=True):
    provenance = observed_provenance(**({"dsp": dsp} if dsp is not None else {}))
    capture = replace(manifest.captures[0], provenance=provenance)
    original = replace(manifest.artifacts[0], frame_interval=FrameInterval(0, 480_000))
    clip = replace(manifest.artifacts[1], frame_interval=FrameInterval(16_000, 112_000))
    audio = original if full else clip
    reference = replace(manifest.artifacts[2], parent_id=audio.artifact_id,
                        start_ms=audio.start_ms, end_ms=audio.end_ms,
                        audio_artifact_id=audio.artifact_id, frame_interval=audio.frame_interval)
    hypothesis = CorpusArtifact("hypothesis-a", capture.capture_id, "hypotheses/a.json", "d" * 64,
                                "stt_hypothesis", audio.start_ms, audio.end_ms,
                                parent_id=audio.artifact_id, audio_artifact_id=audio.artifact_id,
                                frame_interval=audio.frame_interval)
    comparison = CorpusComparison("comparison-a", audio.artifact_id,
                                  reference.artifact_id, hypothesis.artifact_id)
    return replace(manifest, schema_version=2, captures=(capture,),
                   artifacts=(original, clip, reference, hypothesis),
                   comparisons=(comparison,) if comparisons else ())


@pytest.mark.parametrize("full", [False, True])
def test_schema_two_admits_explicit_full_and_preview_comparisons(manifest, full):
    current = aligned_manifest(manifest, full=full)
    assert current.require_comparison("comparison-a", current.reviewed_at) is current.comparisons[0]
    audio, reference, hypothesis = current._comparison_artifacts(current.comparisons[0])
    assert audio.frame_interval == reference.frame_interval == hypothesis.frame_interval
    assert reference.audio_artifact_id == hypothesis.audio_artifact_id == audio.artifact_id
    assert current.captures[0].provenance.stream_generation == 7
    assert current.captures[0].provenance.segment_generation == 23
    assert current.deadline("capture-a") == manifest.deadline("capture-a")


def test_legacy_metadata_is_preserved_but_never_promoted_to_comparable_provenance(manifest):
    assert manifest.schema_version == 1
    assert manifest.captures[0].channel and manifest.captures[0].dsp_revision
    assert manifest.captures[0].provenance is None
    upgraded = replace(manifest, schema_version=2)
    hypothesis = CorpusArtifact("hypothesis-a", "capture-a", "hypotheses/a.json", "d" * 64,
                                "stt_hypothesis", 1000, 7000, parent_id="clip-a",
                                audio_artifact_id="clip-a")
    reference = replace(upgraded.artifacts[2], audio_artifact_id="clip-a")
    with pytest.raises(ValueError, match="comparison_provenance_not_observed"):
        replace(upgraded, artifacts=(*upgraded.artifacts[:2], reference, hypothesis),
                comparisons=(CorpusComparison("comparison-a", "clip-a", "reference-a", "hypothesis-a"),))


@pytest.mark.parametrize("schema", [True, False, 0, 3, "2", 2.0, None])
def test_unknown_or_coerced_schema_never_infers_version_two(manifest, schema):
    with pytest.raises(ValueError, match="unsupported_corpus_schema"):
        replace(manifest, schema_version=schema)


def test_schema_one_rejects_new_alignment_fields_instead_of_ignoring_them(manifest):
    current = aligned_manifest(manifest)
    with pytest.raises(ValueError, match="schema_one_has_no_typed_alignment"):
        replace(current, schema_version=1)


@pytest.mark.parametrize("field,value", [
    ("source", "live_input"), ("lane", "command"),
    ("source_ref", "/dev/usb-private-device"), ("source_ref", ""),
    ("source_ref", "x" * 129), ("source_ref", "input\x00a"),
    ("stream_generation", True), ("stream_generation", -1),
    ("stream_generation", MAX_COUNTER + 1), ("stream_generation", 1.0),
    ("segment_generation", False), ("segment_generation", -1),
    ("segment_generation", MAX_COUNTER + 1),
    ("source_channels", True), ("source_channels", 0), ("source_channels", 65),
    ("selected_channel", False), ("selected_channel", -1), ("selected_channel", 2),
    ("sample_rate_hz", True), ("sample_rate_hz", 7999),
    ("sample_rate_hz", 192_001), ("sample_rate_hz", 16_000.0),
    ("sample_rate_hz", float("nan")), ("sample_rate_hz", float("inf")),
    ("frame_count", True), ("frame_count", 0), ("frame_count", MAX_COUNTER + 1),
    ("dsp", "observed"),
])
def test_capture_provenance_is_closed_typed_and_bounded(field, value):
    with pytest.raises(ValueError):
        observed_provenance(**{field: value})


@pytest.mark.parametrize("certainty,declared,observed", [
    ("observed", "a", "b"),
    (DSPCertainty.UNKNOWN, "a", None),
    (DSPCertainty.UNKNOWN, None, "b"),
    (DSPCertainty.DECLARED, None, None),
    (DSPCertainty.DECLARED, "a", "b"),
    (DSPCertainty.OBSERVED, "a", None),
])
def test_dsp_configuration_cannot_be_serialized_as_an_observation(certainty, declared, observed):
    with pytest.raises(ValueError):
        DSPObservation(certainty, declared, observed)


@pytest.mark.parametrize("dsp", [DSPObservation(), DSPObservation(DSPCertainty.DECLARED, "settings-a")])
def test_unknown_and_declared_dsp_are_metadata_not_a_comparable_cohort(manifest, dsp):
    current = aligned_manifest(manifest, dsp=dsp, comparisons=False)
    assert not current.captures[0].provenance.comparable
    with pytest.raises(ValueError, match="comparison_provenance_not_observed"):
        replace(current, comparisons=(CorpusComparison("comparison-a", "clip-a", "reference-a", "hypothesis-a"),))


def test_observed_dsp_keeps_declared_and_actual_revision_distinct(manifest):
    current = aligned_manifest(manifest, dsp=DSPObservation(DSPCertainty.OBSERVED, "requested-a", "actual-b"))
    dsp = current.captures[0].provenance.dsp
    assert dsp.declared_revision == "requested-a" and dsp.observed_revision == "actual-b"
    with pytest.raises(FrozenInstanceError):
        dsp.observed_revision = "requested-a"


@pytest.mark.parametrize("start,end", [(True, 16), (0, True), (-1, 16), (16, 16),
                                      (17, 16), (0.0, 16), (0, MAX_COUNTER + 1)])
def test_pcm_intervals_reject_boolean_coercion_and_invalid_endpoints(start, end):
    with pytest.raises(ValueError):
        FrameInterval(start, end)


def test_pcm_frame_mismatch_is_not_hidden_by_equal_rounded_millisecond_bounds(manifest):
    current = aligned_manifest(manifest)
    hypothesis = replace(current.artifacts[-1], frame_interval=FrameInterval(16_001, 112_000))
    assert hypothesis.frame_interval.millisecond_bounds(16_000) == (1000, 7000)
    with pytest.raises(ValueError, match="comparison_interval_mismatch"):
        replace(current, artifacts=(*current.artifacts[:-1], hypothesis))


def test_full_segment_reference_cannot_be_used_for_a_preview(manifest):
    current = aligned_manifest(manifest, full=True, comparisons=False)
    comparison = CorpusComparison("comparison-a", "clip-a", "reference-a", "hypothesis-a")
    with pytest.raises(ValueError, match="comparison_interval_mismatch"):
        replace(current, comparisons=(comparison,))


def test_another_audio_with_identical_frames_does_not_rebind_reference_or_hypothesis(manifest):
    current = aligned_manifest(manifest, comparisons=False)
    other_audio = replace(current.artifacts[1], artifact_id="clip-b", relative_path="audio/clip-b.wav", sha256="e" * 64)
    with pytest.raises(ValueError, match="comparison_audio_binding_mismatch"):
        replace(current, artifacts=(*current.artifacts, other_audio),
                comparisons=(CorpusComparison("comparison-a", "clip-b", "reference-a", "hypothesis-a"),))


@pytest.mark.parametrize("kind", ["embedding", "human_reference", "transcript", "copy"])
def test_only_explicit_stt_hypothesis_kind_can_be_compared(manifest, kind):
    current = aligned_manifest(manifest)
    changes = {"kind": kind}
    if kind == "human_reference":
        changes["human_review_ref"] = "review-b"
    changed = replace(current.artifacts[-1], **changes)
    with pytest.raises(ValueError, match="invalid_explicit_audio_binding|invalid_comparison_kind"):
        replace(current, artifacts=(*current.artifacts[:-1], changed))


@pytest.mark.parametrize("which", [2, 3])
def test_reference_and_hypothesis_require_explicit_selected_audio_binding(manifest, which):
    current = aligned_manifest(manifest)
    values = list(current.artifacts)
    values[which] = replace(values[which], audio_artifact_id=None)
    with pytest.raises(ValueError, match="comparison_audio_binding_mismatch"):
        replace(current, artifacts=tuple(values))


def test_audio_binding_cannot_cross_capture_even_with_equal_intervals(manifest):
    current = aligned_manifest(manifest, comparisons=False)
    other_capture = replace(current.captures[0], capture_id="capture-b", recording_sha256="e" * 64,
                            provenance=observed_provenance(selected_channel=0))
    original = replace(current.artifacts[0], artifact_id="original-b", capture_id="capture-b",
                       relative_path="audio/b.wav", sha256="e" * 64)
    reference = replace(current.artifacts[2], parent_id="original-b", audio_artifact_id="original-b")
    with pytest.raises(ValueError, match="invalid_explicit_audio_binding"):
        replace(current, captures=(*current.captures, other_capture),
                artifacts=(*current.artifacts[:2], reference, current.artifacts[3], original))


@pytest.mark.parametrize("event", ["deadline", "revocation", "closure"])
def test_comparison_admission_stops_at_exact_original_or_consent_deadline(manifest, event):
    current = aligned_manifest(manifest)
    deadline = CAPTURED + timedelta(days=30) if event == "deadline" else CAPTURED + timedelta(hours=2)
    if event == "revocation":
        current = replace(current, reviewed_at=deadline,
                          sessions=(replace(current.sessions[0], revoked_at=deadline),))
    elif event == "closure":
        current = replace(current, reviewed_at=deadline, closed_at=deadline)
    else:
        assert current.require_comparison("comparison-a", deadline - timedelta(microseconds=1))
    with pytest.raises(PermissionError, match="comparison_retired"):
        current.require_comparison("comparison-a", deadline)
    assert current.retired_artifacts(deadline) == current.artifacts


def test_new_hypothesis_or_comparison_does_not_reset_original_retention(manifest):
    current = aligned_manifest(manifest)
    revised = replace(current, reviewed_at=CAPTURED + timedelta(days=29))
    assert revised.deadline("capture-a") == current.deadline("capture-a")
    with pytest.raises(PermissionError, match="comparison_retired"):
        revised.require_comparison("comparison-a", CAPTURED + timedelta(days=30))
    with pytest.raises(TypeError):
        replace(revised.comparisons[0], expires_at=CAPTURED + timedelta(days=60))


def test_comparison_is_metadata_only_and_does_not_open_files(manifest, monkeypatch):
    current = aligned_manifest(manifest)
    def forbidden(*args, **kwargs):
        raise AssertionError("pure comparison opened a file")
    monkeypatch.setattr("builtins.open", forbidden)
    assert current.require_comparison("comparison-a", current.reviewed_at).comparison_id == "comparison-a"


def test_comparison_admission_keeps_existing_clock_and_missing_id_failures(manifest):
    current = aligned_manifest(manifest)
    with pytest.raises(ValueError, match="clock_before_review"):
        current.require_comparison("comparison-a", current.reviewed_at - timedelta(microseconds=1))
    with pytest.raises(ValueError, match="timezone_required"):
        current.require_comparison("comparison-a", datetime(2026, 11, 2))
    with pytest.raises(ValueError, match="unknown_comparison"):
        current.require_comparison("absent", current.reviewed_at)


def test_duplicate_or_coerced_comparisons_are_rejected(manifest):
    current = aligned_manifest(manifest)
    with pytest.raises(ValueError, match="duplicate_comparison"):
        replace(current, comparisons=current.comparisons * 2)
    with pytest.raises(ValueError, match="invalid_corpus_comparisons"):
        replace(current, comparisons=list(current.comparisons))
    with pytest.raises(ValueError, match="comparison_artifacts_must_be_distinct"):
        CorpusComparison("comparison-a", "clip-a", "clip-a", "hypothesis-a")


def test_exact_frames_cannot_be_derived_through_an_unknown_parent(manifest):
    current = aligned_manifest(manifest)
    original, clip, reference, hypothesis = current.artifacts
    unknown_parent = replace(clip, frame_interval=None)
    audio = replace(clip, artifact_id="recrop-a", relative_path="audio/recrop.wav", parent_id=clip.artifact_id)
    reference = replace(reference, parent_id=audio.artifact_id, audio_artifact_id=audio.artifact_id)
    hypothesis = replace(hypothesis, parent_id=audio.artifact_id, audio_artifact_id=audio.artifact_id)
    comparison = replace(current.comparisons[0], audio_artifact_id=audio.artifact_id)
    with pytest.raises(ValueError, match="derivative_pcm_parent_unknown"):
        replace(current, artifacts=(original, unknown_parent, audio, reference, hypothesis),
                comparisons=(comparison,))


def test_submillisecond_endpoints_keep_exact_pcm_identity(manifest):
    current = aligned_manifest(manifest)
    provenance = replace(current.captures[0].provenance, frame_count=480_001)
    capture = replace(current.captures[0], duration_ms=30_001, provenance=provenance)
    original = replace(current.artifacts[0], end_ms=30_001, frame_interval=FrameInterval(0, 480_001))
    interval = FrameInterval(16_001, 112_001)
    bounds = interval.millisecond_bounds(provenance.sample_rate_hz)
    assert bounds == (1000, 7001)
    exact = tuple(replace(item, start_ms=bounds[0], end_ms=bounds[1], frame_interval=interval)
                  for item in current.artifacts[1:])
    revised = replace(current, captures=(capture,), artifacts=(original, *exact))
    assert revised.require_comparison("comparison-a", revised.reviewed_at)
    assert revised.artifacts[1].frame_interval == interval


@pytest.mark.parametrize("field,value,error", [
    ("consented_at", CAPTURED + timedelta(minutes=1), "invalid_session_interval"),
    ("split", "reserved", "invalid_split"),
])
def test_comparison_revalidates_session_fields(manifest, field, value, error):
    current = aligned_manifest(manifest)
    object.__setattr__(current.sessions[0], field, value)
    with pytest.raises(ValueError, match=error):
        current.require_comparison("comparison-a", current.reviewed_at)


@pytest.mark.parametrize("records", ["sessions", "captures", "artifacts"])
def test_manifest_rejects_mutable_record_impostors(manifest, records):
    current = aligned_manifest(manifest)
    original = getattr(current, records)
    mutable = SimpleNamespace(**vars(original[0]))
    with pytest.raises(ValueError, match="invalid_corpus_record"):
        replace(current, **{records: (mutable, *original[1:])})


@pytest.mark.parametrize("frames,end_us,admitted", [
    (16_002, 1_000_125, True), (16_002, 1_000_124, False),
    (16_001, 1_000_063, True), (16_001, 1_000_062, False),
])
def test_session_end_uses_exact_pcm_duration_not_rounded_milliseconds(manifest, frames, end_us, admitted):
    current = aligned_manifest(manifest, full=True)
    session = replace(current.sessions[0], ended_at=CAPTURED + timedelta(microseconds=end_us))
    provenance = replace(current.captures[0].provenance, frame_count=frames)
    capture = replace(current.captures[0], duration_ms=1001, provenance=provenance)
    interval = FrameInterval(0, frames)
    artifacts = tuple(replace(item, end_ms=1001, frame_interval=interval)
                      for item in (current.artifacts[0], *current.artifacts[2:]))
    def revised():
        return replace(current, sessions=(session,), captures=(capture,), artifacts=artifacts)
    if admitted:
        assert revised().require_comparison("comparison-a", current.reviewed_at)
    else:
        with pytest.raises(ValueError, match="recording_outside_session"):
            revised()


class SyntheticFold(tzinfo):
    """Two UTC offsets for one synthetic local-hour fold; no tzdb dependency."""

    def utcoffset(self, value):
        return timedelta(hours=-5 if value is not None and value.fold else -4)

    def dst(self, value):
        return timedelta(0)


def folded_manifest(manifest):
    zone = SyntheticFold()
    start = CAPTURED.replace(hour=0, tzinfo=zone)
    session = replace(manifest.sessions[0], consented_at=start - timedelta(minutes=1),
                      started_at=start, ended_at=start.replace(hour=1, fold=0))
    capture = replace(manifest.captures[0], captured_at=start,
                      original_expires_at=start + timedelta(days=30))
    return replace(manifest, sessions=(session,), captures=(capture,),
                   reviewed_at=start.replace(hour=1, minute=30))


def test_folded_wall_time_cannot_accept_consent_after_recording():
    zone = SyntheticFold()
    consent = CAPTURED.replace(hour=1, minute=0, tzinfo=zone, fold=1)
    start = consent.replace(minute=30, fold=0)
    assert consent.astimezone(timezone.utc) > start.astimezone(timezone.utc)
    with pytest.raises(ValueError, match="invalid_session_interval"):
        CorpusSession("s", "p", "c", consent, start, start.replace(hour=3), CorpusSplit.RESERVED)


def test_folded_clock_uses_actual_instant_for_review_and_deadline(manifest):
    current = aligned_manifest(folded_manifest(manifest))
    now = current.reviewed_at.replace(minute=15, fold=1)
    assert now.astimezone(timezone.utc) > current.reviewed_at.astimezone(timezone.utc)
    assert current.require_comparison("comparison-a", now)
    assert current.retired_artifacts(now) == ()


def test_folded_deadline_selects_earliest_actual_event(manifest):
    current = aligned_manifest(folded_manifest(manifest))
    closed = current.reviewed_at.replace(minute=45, fold=0)
    revoked = current.reviewed_at.replace(minute=15, fold=1)
    revised = replace(current, reviewed_at=closed.replace(hour=3, fold=1), closed_at=closed,
                      sessions=(replace(current.sessions[0], revoked_at=revoked),))
    assert revised.deadline("capture-a") == closed.astimezone(timezone.utc)


def test_folded_expiry_cannot_precede_capture_in_utc(manifest):
    capture = folded_manifest(manifest).captures[0]
    captured = capture.captured_at.replace(hour=1, minute=30, fold=1)
    expires = captured.replace(minute=45, fold=0)
    with pytest.raises(ValueError, match="original_deadline_exceeds_consent"):
        replace(capture, captured_at=captured, original_expires_at=expires)


ROOT = Path(__file__).resolve().parents[2]
P6_PROTOCOL_SHA256 = "53a55486e009f549d7a4c42a6bc28ea8239c75a512b8ff50887ec02c54aaba33"
P6_CRITERIA = {
    "C12": (("wer_clear_max_pct", "wer_adverse_max_pct", "coverage_min_pct"),
            ("clear_min", "adverse_min", "critical_condition_min")),
    "C13": (("success_clear_min_pct", "success_adverse_min_pct"), ("family_min",)),
    "C14": (("rejection_max_pct", "false_per_house_hour_max"),
            ("intentional_min", "useful_house_hours_min", "simultaneous_inputs", "review_min_pct")),
    "C15": (("voice_p50_ms_max", "voice_p95_ms_max", "rapid_effect_p95_ms_max"),
            ("simple_runs_min", "family_min")),
    "C16": (("fully_delivered_min_pct", "acoustic_interrupt_p95_ms_max"),
            ("deliveries_min", "physical_interrupts_min")),
    "C17": (("clarity_median_min", "clarity_scale", "severe_max"),
            ("paired_responses_min", "listeners_min")),
    "C18": (("mixed_pairs_max", "pair_precision_min", "pair_recall_min", "clear_coverage_min"),
            ("voices_min", "calibration_clear_per_voice_min", "reserve_clear_per_voice_min",
             "rooms_and_distances_each_min", "per_adverse_condition_min", "first_seed", "last_seed")),
}
P6_FINDINGS = ("AUD-04", "AUD-06", "AUD-07")
P6_TASKS = ("T062", "T071", "T072")


@pytest.fixture(scope="module")
def frozen_protocol():
    document = (ROOT / "docs/plans/p6-evaluation.md").read_bytes()
    assert sha256(document).hexdigest() == P6_PROTOCOL_SHA256
    block, = re.findall(r"(?m)^```json\n([\s\S]*?)^```$", document.decode("utf-8"))
    return json.loads(block)


@pytest.fixture(scope="module")
def live_roadmap():
    return json.loads((ROOT / "docs/plans/2026-11-02-autonomous-roadmap.json").read_text(encoding="utf-8"))


def test_frozen_protocol_pins_evaluators_and_symbols(frozen_protocol):
    assert frozen_protocol["protocol"] == "p6-v1"
    assert set(frozen_protocol["limits"]) == set(P6_CRITERIA)
    for name, pin in frozen_protocol["sources"].items():
        source = (ROOT / name).read_bytes()
        assert sha256(source).hexdigest() == pin["sha256"], name
        tree = ast.parse(source, filename=name)
        for symbol in pin["symbols"]:
            scope = tree
            for segment in symbol.split("."):
                candidates = [node for node in scope.body if getattr(node, "name", None) == segment]
                assert len(candidates) == 1, (name, symbol)
                scope = candidates[0]


def test_p6_roadmap_contract_has_not_drifted(frozen_protocol, live_roadmap):
    def select(section, ids, *fields):
        return {row["id"]: {field: row[field] for field in fields}
                for row in live_roadmap[section] if row["id"] in ids}

    contract = {
        "criteria": select("criteria", P6_CRITERIA, "target", "minimum_evidence", "agreement"),
        "tasks": select("tasks", P6_TASKS, "phase", "criteria", "findings", "human_gate"),
        "findings": select("finding_trace", P6_FINDINGS, "title", "tasks"),
        "stratification": live_roadmap["stratification"],
    }
    assert set(contract["tasks"]["T062"]["criteria"]) == set(P6_CRITERIA)
    assert set(contract["tasks"]["T062"]["findings"]) == set(P6_FINDINGS)
    assert canonical_sha256(contract) == frozen_protocol["roadmap_contract_sha256"]


@pytest.mark.parametrize("criterion", P6_CRITERIA)
def test_frozen_numeric_limits_match_roadmap_text(criterion, frozen_protocol, live_roadmap):
    row, = (item for item in live_roadmap["criteria"] if item["id"] == criterion)
    words = {"cero": 0, "dos": 2, "tres": 3, "cuatro": 4, "diez": 10}
    numbers = re.compile(r"\b(?:cero|dos|tres|cuatro|diez)\b|(?<!\w)\d+(?:\.\d+)?(?!\w)")
    expected = {}
    for field, keys in zip(("target", "minimum_evidence"), P6_CRITERIA[criterion], strict=True):
        values = [words[token] if token in words else float(token) for token in numbers.findall(row[field].lower())]
        expected.update(zip(keys, values, strict=True))
    assert frozen_protocol["limits"][criterion] == expected


@pytest.fixture
def field_inventory():
    calibration = EvaluationClip(
        "clip-cal", "a" * 64, "b" * 64, "session-cal", Partition.CALIBRATION, CAPTURED,
        CAPTURED + timedelta(seconds=4), frozenset({"voice-cal"}), "receipt-cal",
    )
    reserve = EvaluationClip(
        "clip-res", "c" * 64, "d" * 64, "session-res", Partition.RESERVED,
        CAPTURED + timedelta(days=1), CAPTURED + timedelta(days=1, seconds=4),
        frozenset({"voice-res"}), "receipt-res",
    )
    now = CAPTURED + timedelta(days=2)
    receipts = {
        clip.consent_ref: ConsentReceipt(
            clip.session_id, "consented_calibration_and_evaluation", True, clip.speaker_refs,
            CAPTURED - timedelta(minutes=1), now + timedelta(days=1),
        )
        for clip in (calibration, reserve)
    }
    return calibration, reserve, receipts, now


def test_p6_enforces_inventory_but_allows_preregistered_h3_panel(field_inventory):
    calibration, reserve, receipts, now = field_inventory
    validate_inventory((reserve, calibration), receipts, now=now)
    reserve = replace(reserve, speaker_refs=calibration.speaker_refs)
    receipts[reserve.consent_ref] = replace(receipts[reserve.consent_ref], speaker_refs=reserve.speaker_refs)
    with pytest.raises(ValueError, match="cross_partition_speaker"):
        validate_inventory((calibration, reserve), receipts, now=now)
    validate_inventory((calibration, reserve), receipts, now=now, shared_panel=reserve.speaker_refs)

    training = replace(
        calibration, clip_id="clip-train", audio_sha256="e" * 64, source_sha256="f" * 64,
        split=Partition.TRAINING, session_id="session-train", consent_ref="receipt-train",
    )
    receipts[training.consent_ref] = replace(receipts[calibration.consent_ref], session_id=training.session_id)
    with pytest.raises(ValueError, match="cross_partition_speaker"):
        validate_inventory((training, calibration, reserve), receipts, now=now, shared_panel=reserve.speaker_refs)


@pytest.mark.parametrize(
    ("field", "shared_field", "reason"),
    [
        ("clip_id", "clip_id", "clip"),
        ("audio_sha256", "audio_sha256", "audio"),
        ("source_sha256", "source_sha256", "audio"),
        ("audio_sha256", "source_sha256", "audio"),
        ("session_id", "session_id", "session"),
    ],
)
def test_p6_rejects_cross_partition_lineage(field_inventory, field, shared_field, reason):
    calibration, reserve, receipts, now = field_inventory
    contaminated = replace(reserve, **{field: getattr(calibration, shared_field)})
    receipts[reserve.consent_ref] = replace(receipts[reserve.consent_ref], session_id=contaminated.session_id)
    with pytest.raises(ValueError, match=f"cross_partition_{reason}"):
        validate_inventory((contaminated, calibration), receipts, now=now)


def test_p6_rejects_nested_overlapping_windows_but_allows_an_exact_boundary(field_inventory):
    calibration, reserve, receipts, now = field_inventory
    inside = calibration.start + timedelta(seconds=1)
    time_zone = timezone(timedelta(hours=-3))
    reserve = replace(reserve, start=inside.astimezone(time_zone), end=inside + timedelta(seconds=1))
    with pytest.raises(ValueError, match="cross_partition_window"):
        validate_inventory((reserve, calibration), receipts, now=now)
    boundary = replace(reserve, start=calibration.end, end=calibration.end + timedelta(seconds=1))
    validate_inventory((boundary, calibration), receipts, now=now)


def test_p6_rejects_an_empty_split_or_invalid_clip_provenance(field_inventory):
    calibration, reserve, receipts, now = field_inventory
    with pytest.raises(ValueError, match="calibration_and_reserve_required"):
        validate_inventory((calibration,), receipts, now=now)
    with pytest.raises(ValueError, match="invalid_clip_provenance"):
        validate_inventory((calibration, replace(reserve, audio_sha256="")), receipts, now=now)
    with pytest.raises(ValueError, match="duplicate_clip"):
        validate_inventory((calibration, reserve, reserve), receipts, now=now)


@pytest.mark.parametrize(
    "changes",
    [
        None, {"granted": False}, {"granted": 1}, {"purpose": "ordinary_capture"},
        {"session_id": "other-session"}, {"speaker_refs": frozenset()},
        {"granted_at": CAPTURED + timedelta(days=1, seconds=1)},
        {"revoked_at": CAPTURED + timedelta(days=2)},
        {"expires_at": CAPTURED + timedelta(days=2)},
    ],
)
def test_p6_rejects_missing_invalid_or_noncurrent_explicit_consent(field_inventory, changes):
    calibration, reserve, receipts, now = field_inventory
    if changes is None:
        del receipts[reserve.consent_ref]
    else:
        receipts[reserve.consent_ref] = replace(receipts[reserve.consent_ref], **changes)
    with pytest.raises(PermissionError, match="explicit_current_consent_required"):
        validate_inventory((calibration, reserve), receipts, now=now)


def test_p6_seed_order_is_fixed_per_domain_and_independent_of_input_order(frozen_protocol):
    ids = tuple("abcde")
    assert seeded_order(ids, seed=0, purpose="h3") == tuple("aedbc")
    assert seeded_order(ids, seed=9, purpose="h3") == tuple("cadeb")
    assert seeded_order(ids, seed=0, purpose="listening") == tuple("acedb")
    all_ids = tuple(f"clip-{i}" for i in range(20))
    h3 = frozen_protocol["limits"]["C18"]
    orders = {
        seeded_order(all_ids, seed=seed, purpose="h3")
        for seed in range(h3["first_seed"], h3["last_seed"] + 1)
    }
    assert len(orders) == 10
    assert orders == {
        seeded_order(reversed(all_ids), seed=seed, purpose="h3")
        for seed in range(10)
    }
    assert all(sorted(order) == sorted(all_ids) for order in orders)
    with pytest.raises(ValueError, match="unique_ids"):
        seeded_order(("a", "a"), seed=0, purpose="h3")
    with pytest.raises(ValueError, match="unknown_order"):
        seeded_order(ids, seed=10, purpose="h3")


def test_p6_c12_micro_cer_and_coverage_preserve_different_denominators():
    rows = [
        ("¡Café!", "cafe"),
        ("uno dos tres cuatro cinco seis siete ocho nueve", "uno dos tres cuatro cinco seis siete ocho nueve extra"),
        ("luz", ""),
        (None, "luz"),
    ]
    result = recognition(rows)
    assert (result.eligible, result.measured, result.reference_words, result.reference_characters) == (4, 3, 11, 46)
    assert (result.substitutions, result.insertions, result.deletions, result.empty_hypotheses) == (1, 1, 1, 1)
    assert result.wer_micro == pytest.approx(3 / 11)
    assert result.wer_macro == pytest.approx((2 + 1 / 9) / 3)
    assert result.cer_micro == pytest.approx(9 / 46)
    assert result.coverage == 0.75
    historical = build_report(
        [{"reference": ref, "hypothesis": hyp, "vad_prob": 0.9} for ref, hyp in rows[:3]],
        {"0.80-1.00": 3},
    )
    assert historical["confiable"]
    assert historical["wer_reponderado"] == pytest.approx(result.wer_macro)
    assert historical["wer_reponderado"] != pytest.approx(result.wer_micro)
    with pytest.raises(ValueError, match="speech_reference_requires_words"):
        recognition([("habla", "habla"), ("[TV]", None)])
    with pytest.raises(ValueError, match="no_aligned_speech"):
        recognition([("habla", None)])


def test_p6_c13_requires_real_domain_completion_and_independent_effect():
    intent = HouseholdIntentClassifier().classify("recordame comprar leche")
    correct_intent = (intent.family, intent.operation) == (HouseholdFamily.REMINDER, HouseholdOperation.CREATE)
    queued = TaskOutcome(TaskStatus.QUEUED)
    observed = TaskOutcome(TaskStatus.COMPLETED, effect=EffectState.OBSERVED, observation_scope="software_state")
    assert queued.handled and not complete_task(queued, intent_correct=True, targets_correct=True, effect_observed=True)
    assert not complete_task(observed, intent_correct=True, targets_correct=True, effect_observed=False)
    assert not complete_task(observed, intent_correct=True, targets_correct=False, effect_observed=True)
    assert complete_task(observed, intent_correct=correct_intent, targets_correct=True, effect_observed=True)


def test_p6_c14_conservative_bounds_and_post_stt_diagnostic(frozen_protocol):
    limits = frozen_protocol["limits"]["C14"]
    verdict = WakeVerifier(
        WakeVerificationConfig(mode=WakeVerificationMode.ENFORCE, status_path=None), clock=lambda: 1_800_000_000,
    ).evaluate("encendé la luz", room_id="synthetic", wake_score=0.9, wake_vad=None, gate_accept=True)
    assert verdict.code is WakeCheckCode.REJECTED

    counts = WakeCounts(94, 4, 2, 4, 1, 99, 48 * 3600)
    assert counts.review_coverage >= limits["review_min_pct"] / 100
    assert counts.rejection_upper == 0.06 > limits["rejection_max_pct"] / 100
    assert counts.false_per_hour_upper == pytest.approx(5 / 48)
    assert counts.false_per_hour_upper > limits["false_per_house_hour_max"]


def test_p6_c15_and_c16_include_missing_endpoints_and_require_audible_completion():
    samples = [100] * 94 + [1500] + [None] * 5
    assert nearest_rank(samples, 50) == 100
    assert nearest_rank(samples, 95) == 1500
    assert math.isinf(nearest_rank(samples[:-6] + [None] * 6, 95))
    with pytest.raises(ValueError, match="invalid_latency"):
        nearest_rank([-1], 95)

    result = ResponseResult("respuesta", GenerationState.COMPLETE, DeliveryState.COMPLETE, output_started=True)
    assert complete_delivery(result, audible_end_observed=True)
    assert not complete_delivery(result, audible_end_observed=False)
    assert not complete_delivery(replace(result, output_started=False), audible_end_observed=True)
    for state in (DeliveryState.FAILED, DeliveryState.BLOCKED, DeliveryState.TIMED_OUT, DeliveryState.CANCELLED):
        assert not complete_delivery(replace(result, delivery=state), audible_end_observed=True)


def test_p6_c17_participant_medians_are_independent_of_waveform_diagnostics(frozen_protocol):
    limits = frozen_protocol["limits"]["C17"]
    diagnostics = measure_channel(np.array([0.0, 1.0, 1.0, 1.0, 0.0]), 16000)
    assert diagnostics["longest_near_full_scale_run_samples"] == 3
    ratings = ([5, 4, 5], [4, 3, 4], [3, 2, 4])
    assert [clarity_median(scores) >= limits["clarity_median_min"] for scores in ratings] == [True, True, False]
    with pytest.raises(ValueError, match="invalid_clarity_ratings"):
        clarity_median([0, 4])


def test_p6_c18_counts_abstention_as_missed_pairs_and_mixing_across_conditions():
    rows = [
        VoiceAssignment("a1", "a", "group-a", clear=True),
        VoiceAssignment("a2", "a", "group-a", clear=True),
        VoiceAssignment("a3", "a", None, clear=True),
        VoiceAssignment("b1", "b", "group-a", clear=True),
        VoiceAssignment("b2", "b", "single", clear=True),
        VoiceAssignment("c1", "c", "group-a", clear=False),
        VoiceAssignment("a4", "a", None, clear=False, must_abstain=True),
        VoiceAssignment("overlap", None, "single", clear=False, must_abstain=True),
    ]
    counts = voice_pair_counts(rows)
    assert (counts.tp, counts.fp, counts.fn) == (1, 2, 3)
    assert (counts.clear_segments, counts.assigned_clear) == (5, 4)
    assert (counts.precision, counts.recall, counts.coverage) == pytest.approx((1 / 3, 1 / 4, 4 / 5))
    assert (counts.mixed_pairs, counts.abstention_violations) == (5, 1)
    assert voice_pair_counts(reversed(rows)) == counts
    with pytest.raises(ValueError, match="invalid_voice_assignment"):
        voice_pair_counts(rows + [rows[0]])
