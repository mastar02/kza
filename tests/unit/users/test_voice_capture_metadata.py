"""Strict native/JSON review metadata contracts with synthetic PCM only."""

from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from src.evaluation.provenance import (
    CaptureLane, CaptureProvenance, CaptureSource, DSPCertainty, DSPObservation,
    FrameInterval,
)
from src.users.voice_capture import VoiceCaptureReview, build_review_capture
import src.users.voice_capture_metadata as metadata
from src.users.voice_capture_metadata import (
    decode_audio_metadata, decode_transcript_metadata,
    encode_audio_metadata, encode_transcript_metadata,
)


FRAME_COUNT = 160_001
PRIVATE_MARKER = "SYNTHETIC_PRIVATE_PARSE_CONTENT"


def provenance(*, channel=0, lane=CaptureLane.COMMAND, count=FRAME_COUNT, **changes):
    values = dict(source=CaptureSource.LIVE_INPUT, source_ref="synthetic-input",
                  lane=lane, stream_generation=7, segment_generation=23,
                  source_channels=2, selected_channel=channel,
                  sample_rate_hz=16_000, frame_count=count,
                  dsp=DSPObservation(DSPCertainty.DECLARED, "synthetic-settings"))
    values.update(changes)
    return CaptureProvenance(**values)


def audio_document():
    return json.loads(encode_audio_metadata(provenance(), FrameInterval(0, 96_000)))


def text_document():
    return json.loads(encode_transcript_metadata(provenance(), FrameInterval(0, FRAME_COUNT), False))


def invalid(function, *args):
    with pytest.raises(ValueError) as caught:
        function(*args)
    error = caught.value
    assert error.__context__ is None, "a fixed error kept a private parser/validator exception"
    assert error.__cause__ is None
    assert PRIVATE_MARKER not in str(error)
    assert 0 < len(str(error)) < 128
    return error


def mutate_document(document, path, value):
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return json.dumps(document, ensure_ascii=False, allow_nan=True)


def test_native_audio_and_text_roundtrip_keep_exact_frames_and_distinct_dsp_certainty():
    observed = provenance()
    audio_span, text_span = FrameInterval(0, 96_000), FrameInterval(0, FRAME_COUNT)
    audio_raw = encode_audio_metadata(observed, audio_span)
    text_raw = encode_transcript_metadata(observed, text_span, False)
    assert type(audio_raw) is str and type(text_raw) is str
    assert len(audio_raw.encode("utf-8")) <= 4096
    assert decode_audio_metadata(audio_raw) == (observed, audio_span)
    assert decode_transcript_metadata(text_raw) == (observed, text_span, False)
    restored, _ = decode_audio_metadata(audio_raw)
    assert restored.dsp.certainty is DSPCertainty.DECLARED
    assert restored.dsp.observed_revision is None and not restored.comparable
    assert type(restored) is CaptureProvenance and type(restored.dsp) is DSPObservation


def test_wire_v1_has_only_the_closed_document_and_native_fields():
    audio, text = audio_document(), text_document()
    assert set(audio) == {"schema_version", "provenance", "span"}
    assert set(text) == {"schema_version", "provenance", "span", "text_truncated"}
    assert audio["schema_version"] == text["schema_version"] == 1
    assert set(audio["provenance"]) == {
        "source", "source_ref", "lane", "stream_generation", "segment_generation",
        "source_channels", "selected_channel", "sample_rate_hz", "frame_count", "dsp",
    }
    assert set(audio["provenance"]["dsp"]) == {"certainty", "declared_revision", "observed_revision"}
    assert set(audio["span"]) == {"start_frame", "end_frame"}


def test_sql_null_is_all_unknown_without_inventing_frames_or_truncation():
    assert encode_audio_metadata(None, None) is None
    assert decode_audio_metadata(None) == (None, None)
    assert encode_transcript_metadata(None, None, None) is None
    assert decode_transcript_metadata(None) == (None, None, None)


@pytest.mark.parametrize("truncated", [False, True])
def test_text_can_preserve_an_observed_truncation_flag_without_audio_provenance(truncated):
    raw = encode_transcript_metadata(None, None, truncated)
    assert type(raw) is str
    assert decode_transcript_metadata(raw) == (None, None, truncated)
    assert json.loads(raw) == {"schema_version": 1, "provenance": None,
                               "span": None, "text_truncated": truncated}


@pytest.mark.parametrize("pair", [(None, FrameInterval(0, 96_000)), (provenance(), None)])
def test_partial_native_pairs_are_never_silently_completed(pair):
    invalid(encode_audio_metadata, *pair)
    invalid(encode_transcript_metadata, *pair, False)


def test_nonnull_audio_cannot_claim_an_unknown_pair():
    invalid(decode_audio_metadata, json.dumps({"schema_version": 1, "provenance": None, "span": None}))


@pytest.mark.parametrize("value", [0, 1, "false", "true", [], {}])
def test_truncation_is_exact_boolean_or_unknown(value):
    invalid(encode_transcript_metadata, provenance(), FrameInterval(0, FRAME_COUNT), value)
    invalid(decode_transcript_metadata, mutate_document(text_document(), ("text_truncated",), value))


@pytest.mark.parametrize("decoder,factory", [(decode_audio_metadata, audio_document),
                                            (decode_transcript_metadata, text_document)])
@pytest.mark.parametrize("version", [True, False, 0, 2, "1", 1.0, None])
def test_unknown_or_coerced_wire_versions_fail_closed(decoder, factory, version):
    invalid(decoder, mutate_document(factory(), ("schema_version",), version))


@pytest.mark.parametrize("decoder,factory", [(decode_audio_metadata, audio_document),
                                            (decode_transcript_metadata, text_document)])
@pytest.mark.parametrize("path", [("extra",), ("provenance", "extra"),
                                  ("provenance", "dsp", "extra"), ("span", "extra")])
def test_extra_fields_at_every_level_are_rejected(decoder, factory, path):
    invalid(decoder, mutate_document(factory(), path, PRIVATE_MARKER))


@pytest.mark.parametrize("path", [
    ("provenance", "source"), ("provenance", "lane"),
    ("provenance", "stream_generation"), ("provenance", "segment_generation"),
    ("provenance", "source_channels"), ("provenance", "selected_channel"),
    ("provenance", "sample_rate_hz"), ("provenance", "frame_count"),
    ("provenance", "dsp", "certainty"), ("provenance", "dsp", "declared_revision"),
    ("provenance", "dsp", "observed_revision"), ("span", "start_frame"), ("span", "end_frame"),
])
def test_missing_native_fields_do_not_infer_defaults(path):
    document = audio_document()
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    invalid(decode_audio_metadata, json.dumps(document))


@pytest.mark.parametrize("raw", [
    '{"schema_version":1,"schema_version":1,"provenance":null,"span":null}',
    '{"schema_version":1,"provenance":{"source":"live_input","source":"live_input"},"span":null}',
    '{"schema_version":1,"provenance":null,"span":{"start_frame":0,"start_frame":0,"end_frame":1}}',
    '{"schema_version":1,"provenance":null,"span":null,"text_truncated":false,"text_truncated":false}',
])
def test_duplicate_keys_are_rejected_even_when_values_are_equal(raw):
    invalid(decode_audio_metadata, raw)
    invalid(decode_transcript_metadata, raw)


@pytest.mark.parametrize("path,value", [
    (("provenance", "stream_generation"), True),
    (("provenance", "segment_generation"), -1),
    (("provenance", "selected_channel"), 2),
    (("provenance", "sample_rate_hz"), 16_000.0),
    (("provenance", "frame_count"), False),
    (("provenance", "source"), "unknown-source"),
    (("provenance", "lane"), "unknown-lane"),
    (("provenance", "dsp", "certainty"), "configured"),
    (("provenance", "dsp", "observed_revision"), "not-observed"),
    (("span", "start_frame"), True), (("span", "end_frame"), FRAME_COUNT + 1),
])
def test_native_validation_is_repeated_after_json_decode(path, value):
    invalid(decode_audio_metadata, mutate_document(audio_document(), path, value))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_numbers_are_rejected_without_parser_context(constant):
    raw = json.dumps(audio_document()).replace('"end_frame": 96000', '"end_frame": '+constant)
    invalid(decode_audio_metadata, raw)


@pytest.mark.parametrize("raw", [b"{}", bytearray(b"{}"), {}, [], 1, True, "[]", "null", ""])
def test_decoder_requires_sql_null_or_json_object_text(raw):
    invalid(decode_audio_metadata, raw)
    invalid(decode_transcript_metadata, raw)


def test_utf8_byte_limit_precedes_json_parsing(monkeypatch):
    raw = json.dumps({"schema_version": 1, "provenance": None, "span": None,
                      "text_truncated": False, "padding": "é" * 2100}, ensure_ascii=False)
    assert len(raw) < 4096 < len(raw.encode("utf-8"))
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("oversize input reached the JSON parser")
    monkeypatch.setattr(metadata.json, "loads", forbidden)
    invalid(decode_audio_metadata, raw)
    invalid(decode_transcript_metadata, raw)
    assert calls == []


def test_expected_decode_errors_are_fixed_and_drop_private_parse_frames():
    malformed = '{"private":"'+PRIVATE_MARKER+'",'
    invalid_shape = json.dumps({"schema_version": 1, "provenance": PRIVATE_MARKER, "span": None})
    first = invalid(decode_audio_metadata, malformed)
    second = invalid(decode_audio_metadata, invalid_shape)
    assert str(first) == str(second)
    for error in (first, second):
        traceback = error.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if frame.f_globals.get("__name__") == metadata.__name__:
                assert PRIVATE_MARKER not in repr(frame.f_locals)
            traceback = traceback.tb_next


def test_frozen_native_objects_are_revalidated_before_encoding():
    observed = provenance()
    object.__setattr__(observed, "selected_channel", True)
    invalid(encode_audio_metadata, observed, FrameInterval(0, 96_000))
    span = FrameInterval(0, 96_000)
    object.__setattr__(span, "end_frame", True)
    invalid(encode_audio_metadata, provenance(), span)
    invalid(encode_audio_metadata, SimpleNamespace(**vars(provenance())), FrameInterval(0, 96_000))
    invalid(encode_transcript_metadata, provenance(), SimpleNamespace(start_frame=0, end_frame=FRAME_COUNT), False)


def test_existing_three_argument_builder_keeps_all_new_observations_unknown():
    result = build_review_capture(np.zeros(16_000, dtype=np.float32), "Synthetic full segment", "Synthetic room")
    assert type(result) is VoiceCaptureReview
    assert result.audio_provenance is result.audio_span is None
    assert result.transcript_provenance is result.transcript_span is result.transcript_truncated is None
    assert result.duration_s == result.clip_duration_s == 1.


def test_builder_anchors_audio_frames_at_owned_pcm_zero_without_shortening_text():
    audio = np.zeros(FRAME_COUNT, dtype=np.float32)
    text = "Synthetic transcript of the complete segment, never inferred from its preview."
    observed = provenance()
    result = build_review_capture(audio, text, None, audio_provenance=observed,
        transcript_provenance=observed, transcript_span=FrameInterval(0, FRAME_COUNT), transcript_truncated=False)
    assert result.audio_provenance == observed and result.audio_span == FrameInterval(0, 96_000)
    assert result.transcript_span == FrameInterval(0, FRAME_COUNT)
    assert result.transcript == text and result.transcript_truncated is False
    assert result.duration_s == FRAME_COUNT / 16_000 and result.clip_duration_s == 6.
    assert result.audio_provenance.dsp.certainty is DSPCertainty.DECLARED


def test_ambient_audio_and_transcript_keep_independent_selected_channels():
    audio_source = provenance(channel=0, lane=CaptureLane.AMBIENT)
    stt_source = provenance(channel=1, lane=CaptureLane.AMBIENT)
    result = build_review_capture(np.zeros(FRAME_COUNT, dtype=np.float32), "Synthetic ambient STT", None,
        audio_provenance=audio_source, transcript_provenance=stt_source,
        transcript_span=FrameInterval(0, FRAME_COUNT), transcript_truncated=False)
    assert result.audio_provenance.selected_channel == 0
    assert result.transcript_provenance.selected_channel == 1
    assert result.audio_provenance != result.transcript_provenance


def test_one_sample_past_millisecond_boundary_survives_native_and_json_roundtrip():
    count = 12_801
    observed = provenance(count=count)
    result = build_review_capture(np.zeros(count, dtype=np.float32), "Synthetic short segment", None,
                                 audio_provenance=observed)
    assert result.audio_span == FrameInterval(0, count)
    assert result.duration_s == result.clip_duration_s == count / 16_000
    assert decode_audio_metadata(encode_audio_metadata(result.audio_provenance, result.audio_span)) == (observed, result.audio_span)


@pytest.mark.parametrize("changes", [{"sample_rate_hz": 8000}, {"frame_count": FRAME_COUNT - 1},
                                     {"frame_count": FRAME_COUNT + 1}])
def test_builder_rejects_metadata_that_does_not_describe_actual_full_pcm(changes):
    with pytest.raises(ValueError):
        build_review_capture(np.zeros(FRAME_COUNT, dtype=np.float32), "Synthetic text", None,
                             audio_provenance=provenance(**changes))


@pytest.mark.parametrize("span", [FrameInterval(1, FRAME_COUNT), FrameInterval(0, 96_000),
                                 FrameInterval(0, FRAME_COUNT + 1)])
def test_full_segment_text_never_adopts_a_preview_or_shifted_span(span):
    with pytest.raises(ValueError):
        build_review_capture(np.zeros(FRAME_COUNT, dtype=np.float32), "Synthetic full text", None,
            transcript_provenance=provenance(), transcript_span=span, transcript_truncated=False)


def test_native_review_rejects_wav_span_and_subframe_duration_disagreements():
    result = build_review_capture(np.zeros(FRAME_COUNT, dtype=np.float32), "Synthetic text", None,
                                 audio_provenance=provenance())
    with pytest.raises(ValueError):
        replace(result, audio_span=FrameInterval(0, 95_999))
    with pytest.raises(ValueError):
        replace(result, clip_duration_s=result.clip_duration_s - 1 / 16_000)
    # A round(duration*rate) check alone would incorrectly accept this change.
    with pytest.raises(ValueError):
        replace(result, duration_s=result.duration_s + .25 / 16_000)
    damaged = bytearray(result.audio_wav)
    damaged[0:4] = b"FAIL"
    with pytest.raises(ValueError):
        replace(result, audio_wav=bytes(damaged))


def test_text_unknown_pair_can_still_record_that_storage_truncated_the_text():
    result = build_review_capture(np.zeros(16_000, dtype=np.float32), "Synthetic text", None,
                                 transcript_truncated=True)
    assert result.transcript_provenance is result.transcript_span is None
    assert result.transcript_truncated is True


def test_review_metadata_does_not_appear_in_repr_or_gain_any_source_or_audio_payload():
    result = build_review_capture(np.zeros(FRAME_COUNT, dtype=np.float32), "Synthetic private transcript", None,
                                 audio_provenance=provenance())
    assert "Synthetic private transcript" not in repr(result)
    document = audio_document()
    assert not {"audio_wav", "transcript", "sources", "source_digests", "person_id", "room_id"} & set(document)
