"""Independent modality metadata and bounded restored-data reads on synthetic databases."""

import asyncio
from base64 import b64encode
import json
import sqlite3
from struct import pack
import threading

import pytest

from src.dashboard.authority import EvaluationModality
from src.dashboard.evaluation_grants import EvaluationGrantSelection
from src.evaluation.provenance import (
    CaptureLane, CaptureProvenance, CaptureSource, DSPCertainty, DSPObservation,
    FrameInterval,
)
from src.users.voice_capture_metadata import encode_audio_metadata, encode_transcript_metadata
from src.users.voice_evaluation import EvaluationProjection
from src.users.voice_ledger import VoiceReviewer
from tests.integration import test_voice_evaluation_reader as native


pytestmark = pytest.mark.asyncio
evaluation_graph = native.evaluation_graph
PRIVATE_MARKER = "SYNTHETIC_RESTORED_PRIVATE_CANARY"
READERS = ("src.users.voice_ledger", "src.users.voice_evaluation")


def reviewer(g):
    return VoiceReviewer.from_dashboard(g.root.principal,
        current=lambda:g.root.current(frozenset({"voices:read", "voices:review"})))


async def operator_read(g):
    return await g.voices.acall(g.voices.ledger.review_capture, g.identity, reviewer(g))


async def seed_metadata(g, *, audio_known=True, text_known=True, truncated=False):
    # The fixture's owned recording is three seconds, not a cropped preview.
    assert g.review.duration_s == g.review.clip_duration_s
    frames = (len(g.review.audio_wav) - 44) // 2
    audio = CaptureProvenance(CaptureSource.LIVE_INPUT, "synthetic-audio-window",
        CaptureLane.AMBIENT, 7, 23, 2, 0, 16_000, frames,
        DSPObservation(DSPCertainty.DECLARED, "synthetic-settings"))
    text = CaptureProvenance(CaptureSource.LIVE_INPUT, "synthetic-stt-window",
        CaptureLane.AMBIENT, 7, 23, 2, 1, 16_000, frames,
        DSPObservation(DSPCertainty.UNKNOWN))
    raw_audio = encode_audio_metadata(audio, FrameInterval(0, frames)) if audio_known else None
    raw_text = (encode_transcript_metadata(text, FrameInterval(0, frames), truncated)
        if text_known else encode_transcript_metadata(None, None, truncated))
    await native.sql(g, "UPDATE voice_capture_reviews SET audio_metadata_json=?,transcript_metadata_json=? WHERE fingerprint_id=?",
                     (raw_audio, raw_text, g.identity))
    return raw_audio, raw_text


async def selected_read(g, actor, reads):
    allowed = {
        "facts": {"id", "revision", "expires_at", "state"},
        "conversation_tombstones": {"conversation_id", "deleted_at"},
        "voice_fingerprints": {"id", "sources_json", "expires_at"},
        "voice_capture_reviews": {"fingerprint_id"} | (
            {"audio_wav", "duration_s", "clip_duration_s", "audio_metadata_json"}
            if actor.modality is EvaluationModality.AUDIO_PREVIEW else
            {"transcript", "transcript_metadata_json"}),
    }
    def authorize(operation, table, column, *_):
        if operation == sqlite3.SQLITE_READ:
            reads.append((table, column))
            if table not in allowed or column not in allowed[table]:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    def read():
        db = g.privacy.ledger._db
        db.set_authorizer(authorize)
        try:
            return g.voices.evaluation.read(actor)
        finally:
            db.set_authorizer(None)
    return await g.voices.acall(read)


async def no_selected_row_materialized(g, function, args, watched_field):
    """A WHERE rejection must occur before Python receives the private row."""
    materialized = []
    def call():
        db = g.privacy.ledger._db
        previous = db.row_factory
        def factory(cursor, values):
            fields = {item[0] for item in cursor.description}
            if watched_field in fields:
                materialized.append(True)
                raise AssertionError("invalid restored row crossed the SQL fetch boundary")
            return sqlite3.Row(cursor, values)
        db.row_factory = factory
        try:
            return function(*args)
        finally:
            db.row_factory = previous
    with pytest.raises(LookupError):
        await g.voices.acall(call)
    assert materialized == []


def assert_private_error_cleared(error):
    assert error.__context__ is None and error.__cause__ is None
    assert PRIVATE_MARKER not in str(error)
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_globals.get("__name__") in READERS:
            assert PRIVATE_MARKER not in repr(frame.f_locals)
        traceback = traceback.tb_next


async def test_operator_returns_fresh_canonical_independent_metadata_without_raw_alias(evaluation_graph):
    g = evaluation_graph
    raw_audio, raw_text = await seed_metadata(g, truncated=True)
    first = await operator_read(g)
    assert first["audio_metadata"] == json.loads(raw_audio)
    assert first["transcript_metadata"] == json.loads(raw_text)
    assert first["audio_metadata"]["provenance"]["selected_channel"] == 0
    assert first["transcript_metadata"]["provenance"]["selected_channel"] == 1
    assert first["transcript_metadata"]["text_truncated"] is True
    first["audio_metadata"]["provenance"]["dsp"]["declared_revision"] = "synthetic-mutated"
    second = await operator_read(g)
    assert second["audio_metadata"] == json.loads(raw_audio)
    assert second["audio_metadata"] is not first["audio_metadata"]


async def test_legacy_null_and_known_truncation_with_unknown_pair_are_distinct(evaluation_graph):
    g = evaluation_graph
    first = await operator_read(g)
    assert first["audio_metadata"] is first["transcript_metadata"] is None
    await seed_metadata(g, audio_known=False, text_known=False, truncated=True)
    second = await operator_read(g)
    assert second["audio_metadata"] is None
    assert second["transcript_metadata"] == {"schema_version": 1, "provenance": None,
                                            "span": None, "text_truncated": True}


@pytest.mark.parametrize("modality", list(EvaluationModality))
async def test_selected_metadata_uses_only_its_granted_columns_and_payload(evaluation_graph, modality):
    g = evaluation_graph
    raw_audio, raw_text = await seed_metadata(g)
    issued = await g.issue(modality)
    if modality is EvaluationModality.AUDIO_PREVIEW:
        await native.sql(g, "UPDATE voice_capture_reviews SET transcript=?,transcript_metadata_json=? WHERE fingerprint_id=?",
                         (PRIVATE_MARKER * 1000, "malformed-other-modality", g.identity))
    else:
        await native.sql(g, "UPDATE voice_capture_reviews SET audio_wav=?,audio_metadata_json=?,duration_s=? WHERE fingerprint_id=?",
                         (b"malformed-other-modality", "malformed-other-modality", PRIVATE_MARKER * 1000, g.identity))
    reads = []
    projection = await selected_read(g, issued.actor, reads)
    row = projection.rows[0]
    if modality is EvaluationModality.AUDIO_PREVIEW:
        assert set(row) == {"id", "modality", "audio_mime", "audio_base64", "clip_duration_s", "audio_metadata"}
        assert row["audio_metadata"] == json.loads(raw_audio)
        assert not {"transcript", "transcript_metadata_json", "room_id"} & {column for table, column in reads if table == "voice_capture_reviews"}
    else:
        assert set(row) == {"id", "modality", "transcript_scope", "transcript", "transcript_metadata"}
        assert row["transcript_metadata"] == json.loads(raw_text)
        assert not {"audio_wav", "duration_s", "clip_duration_s", "audio_metadata_json", "room_id"} & {column for table, column in reads if table == "voice_capture_reviews"}
    assert projection.source_digests == g.digests and projection.revision.current()


@pytest.mark.parametrize("column,value", [
    ("transcript", "\x00" + "x" * 40_000),
    ("transcript", sqlite3.Binary(b"synthetic-wrong-storage")),
    ("room_id", "\x00" + "x" * 512),
    ("room_id", sqlite3.Binary(b"synthetic-wrong-storage")),
    ("audio_wav", "x" * 25644),
    ("audio_wav", sqlite3.Binary(b"x" * 192_045)),
    ("duration_s", PRIVATE_MARKER * 1000),
    ("clip_duration_s", PRIVATE_MARKER * 1000),
    ("duration_s", float("inf")),
    ("audio_metadata_json", " " * 4097),
    ("audio_metadata_json", sqlite3.Binary(b"{}")),
    ("transcript_metadata_json", sqlite3.Binary(b"{}")),
])
async def test_operator_rejects_restored_types_and_bytes_before_python_fetch(evaluation_graph, column, value):
    g = evaluation_graph
    assert column in {"transcript", "room_id", "audio_wav", "duration_s", "clip_duration_s",
                      "audio_metadata_json", "transcript_metadata_json"}
    await native.sql(g, f"UPDATE voice_capture_reviews SET {column}=? WHERE fingerprint_id=?", (value, g.identity))
    await no_selected_row_materialized(g, g.voices.ledger.review_capture, (g.identity, reviewer(g)), "audio_wav")


@pytest.mark.parametrize("modality,column,value,watched", [
    (EvaluationModality.AUDIO_PREVIEW, "audio_wav", "x" * 25644, "audio_wav"),
    (EvaluationModality.AUDIO_PREVIEW, "audio_metadata_json", " " * 4097, "audio_wav"),
    (EvaluationModality.AUDIO_PREVIEW, "duration_s", PRIVATE_MARKER * 1000, "audio_wav"),
    (EvaluationModality.TRANSCRIPT_FULL, "transcript", "\x00" + "x" * 40_000, "transcript"),
    (EvaluationModality.TRANSCRIPT_FULL, "transcript", sqlite3.Binary(b"synthetic-BLOB"), "transcript"),
    (EvaluationModality.TRANSCRIPT_FULL, "transcript_metadata_json", sqlite3.Binary(b"{}"), "transcript"),
])
async def test_selected_evaluation_bounds_precede_fetch(evaluation_graph, modality, column, value, watched):
    g = evaluation_graph
    issued = await g.issue(modality)
    await native.sql(g, f"UPDATE voice_capture_reviews SET {column}=? WHERE fingerprint_id=?", (value, g.identity))
    await no_selected_row_materialized(g, g.voices.evaluation.read, (issued.actor,), watched)


@pytest.mark.parametrize("column,value", [
    ("sources_json", "\x00" + "x" * 262_144),
    ("sources_json", sqlite3.Binary(b"[]")),
    ("expires_at", PRIVATE_MARKER * 1000),
    ("expires_at", float("inf")),
])
async def test_common_source_fields_are_bounded_before_any_reader_fetch(evaluation_graph, column, value):
    g = evaluation_graph
    issued = await g.issue(EvaluationModality.AUDIO_PREVIEW)
    await native.sql(g, f"UPDATE voice_fingerprints SET {column}=? WHERE id=?", (value, g.identity))
    await no_selected_row_materialized(g, g.voices.ledger.review_capture, (g.identity, reviewer(g)), "sources_json")
    await no_selected_row_materialized(g, g.voices.evaluation.read, (issued.actor,), "sources_json")


@pytest.mark.parametrize("field", ["sources_json", "transcript", "room_id", "audio_metadata_json", "transcript_metadata_json"])
async def test_operator_invalid_utf8_has_fixed_error_without_private_database_context(evaluation_graph, field):
    g = evaluation_graph
    canary = PRIVATE_MARKER.encode() + b"\xff"
    if field == "sources_json":
        await native.sql(g, "UPDATE voice_fingerprints SET sources_json=CAST(? AS TEXT) WHERE id=?", (canary, g.identity))
    else:
        await native.sql(g, f"UPDATE voice_capture_reviews SET {field}=CAST(? AS TEXT) WHERE fingerprint_id=?", (canary, g.identity))
    with pytest.raises(LookupError, match="voice_capture_unavailable") as caught:
        await operator_read(g)
    assert_private_error_cleared(caught.value)


@pytest.mark.parametrize("modality,column", [
    (EvaluationModality.AUDIO_PREVIEW, "audio_metadata_json"),
    (EvaluationModality.TRANSCRIPT_FULL, "transcript_metadata_json"),
    (EvaluationModality.TRANSCRIPT_FULL, "transcript"),
])
async def test_selected_evaluation_invalid_utf8_does_not_retain_decode_error(evaluation_graph, modality, column):
    g = evaluation_graph
    issued = await g.issue(modality)
    await native.sql(g, f"UPDATE voice_capture_reviews SET {column}=CAST(? AS TEXT) WHERE fingerprint_id=?",
                     (PRIVATE_MARKER.encode() + b"\xff", g.identity))
    with pytest.raises(LookupError, match="evaluation_sample_unavailable") as caught:
        await selected_read(g, issued.actor, [])
    assert_private_error_cleared(caught.value)


@pytest.mark.parametrize("text", ["", "\x00" + "x" * 8191, "😀" * 8192])
async def test_bounded_text_preserves_nul_and_four_byte_unicode(evaluation_graph, text):
    g = evaluation_graph
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    await native.sql(g, "UPDATE voice_capture_reviews SET transcript=? WHERE fingerprint_id=?", (text, g.identity))
    assert (await operator_read(g))["transcript"] == text
    assert (await selected_read(g, issued.actor, [])).rows[0]["transcript"] == text


async def test_character_limit_is_applied_after_bounded_decode(evaluation_graph):
    g = evaluation_graph
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    await native.sql(g, "UPDATE voice_capture_reviews SET transcript=? WHERE fingerprint_id=?", ("x" * 8193, g.identity))
    with pytest.raises(LookupError):
        await operator_read(g)
    with pytest.raises(LookupError):
        await selected_read(g, issued.actor, [])


async def test_room_byte_and_character_boundaries_preserve_unknown(evaluation_graph):
    g = evaluation_graph
    for room in (None, "😀" * 128):
        await native.sql(g, "UPDATE voice_capture_reviews SET room_id=? WHERE fingerprint_id=?", (room, g.identity))
        assert (await operator_read(g))["room_id"] == room
    await native.sql(g, "UPDATE voice_capture_reviews SET room_id=? WHERE fingerprint_id=?", ("x" * 129, g.identity))
    with pytest.raises(LookupError):
        await operator_read(g)


@pytest.mark.parametrize("modality,key,column", [
    (EvaluationModality.AUDIO_PREVIEW, "audio_metadata", "audio_metadata_json"),
    (EvaluationModality.TRANSCRIPT_FULL, "transcript_metadata", "transcript_metadata_json"),
])
async def test_exact_metadata_byte_boundary_and_canonical_output(evaluation_graph, modality, key, column):
    g = evaluation_graph
    audio, text = await seed_metadata(g)
    raw = audio if modality is EvaluationModality.AUDIO_PREVIEW else text
    padded = raw + " " * (4096 - len(raw.encode("utf-8")))
    assert len(padded.encode("utf-8")) == 4096
    issued = await g.issue(modality)
    await native.sql(g, f"UPDATE voice_capture_reviews SET {column}=? WHERE fingerprint_id=?", (padded, g.identity))
    assert (await selected_read(g, issued.actor, [])).rows[0][key] == json.loads(raw)
    assert (await operator_read(g))[key] == json.loads(raw)


@pytest.mark.parametrize("modality,column,raw", [
    (EvaluationModality.AUDIO_PREVIEW, "audio_metadata_json", "{}"),
    (EvaluationModality.AUDIO_PREVIEW, "audio_metadata_json", "7"),
    (EvaluationModality.AUDIO_PREVIEW, "audio_metadata_json", '{"schema_version":1,"schema_version":1,"provenance":null,"span":null}'),
    (EvaluationModality.TRANSCRIPT_FULL, "transcript_metadata_json", '{"schema_version":1,"provenance":null,"span":null,"text_truncated":null}'),
])
async def test_malformed_nonnull_metadata_never_downgrades_to_legacy(evaluation_graph, modality, column, raw):
    g = evaluation_graph
    issued = await g.issue(modality)
    await native.sql(g, f"UPDATE voice_capture_reviews SET {column}=? WHERE fingerprint_id=?", (raw, g.identity))
    with pytest.raises(LookupError):
        await operator_read(g)
    with pytest.raises(LookupError):
        await selected_read(g, issued.actor, [])


async def test_text_metadata_never_adopts_a_preview_interval(evaluation_graph):
    g = evaluation_graph
    _, raw_text = await seed_metadata(g)
    document = json.loads(raw_text)
    document["span"]["end_frame"] -= 1
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    await native.sql(g, "UPDATE voice_capture_reviews SET transcript_metadata_json=? WHERE fingerprint_id=?",
                     (json.dumps(document), g.identity))
    with pytest.raises(LookupError):
        await selected_read(g, issued.actor, [])


@pytest.mark.parametrize("modality", list(EvaluationModality))
async def test_projection_revalidates_closed_modality_metadata_and_drops_native_aliases(evaluation_graph, modality):
    g = evaluation_graph
    await seed_metadata(g)
    issued = await g.issue(modality)
    projection = await selected_read(g, issued.actor, [])
    key = "audio_metadata" if modality is EvaluationModality.AUDIO_PREVIEW else "transcript_metadata"
    old = projection.rows[0][key]
    EvaluationProjection.__post_init__(projection)
    assert projection.rows[0][key] == old and projection.rows[0][key] is not old
    old["provenance"]["source_ref"] = "synthetic-alias-changed"
    assert projection.rows[0][key]["provenance"]["source_ref"] != old["provenance"]["source_ref"]
    projection.rows[0][key]["extra"] = True
    with pytest.raises(ValueError):
        EvaluationProjection.__post_init__(projection)


async def test_projection_revalidates_actual_audio_frames_not_only_base64_shape(evaluation_graph):
    g = evaluation_graph
    await seed_metadata(g)
    issued = await g.issue(EvaluationModality.AUDIO_PREVIEW)
    projection = await selected_read(g, issued.actor, [])
    projection.rows[0]["clip_duration_s"] += 1 / 16_000
    with pytest.raises(ValueError):
        EvaluationProjection.__post_init__(projection)


def projection_audio_payload(full_frames, metadata):
    """Synthetic canonical PCM preview with an independent full-input count."""
    preview_frames = min(full_frames, 96000)
    pcm = b"\0\0" * preview_frames
    header = pack("<4sI4s4sIHHIIHH4sI", b"RIFF", len(pcm) + 36, b"WAVE",
                  b"fmt ", 16, 1, 1, 16000, 32000, 2, 16, b"data", len(pcm))
    metadata = json.loads(json.dumps(metadata))
    metadata["provenance"]["frame_count"] = full_frames
    metadata["span"] = {"start_frame": 0, "end_frame": preview_frames}
    return {
        "audio_base64": b64encode(header + pcm).decode("ascii"),
        "clip_duration_s": preview_frames / 16000,
        "audio_metadata": metadata,
    }


@pytest.mark.parametrize("full_frames", [12800, 480000])
async def test_projection_accepts_exact_audio_full_frame_bounds(evaluation_graph, full_frames):
    g = evaluation_graph
    await seed_metadata(g)
    issued = await g.issue(EvaluationModality.AUDIO_PREVIEW)
    projection = await selected_read(g, issued.actor, [])
    row = projection.rows[0]
    row.update(projection_audio_payload(full_frames, row["audio_metadata"]))
    EvaluationProjection.__post_init__(projection)
    assert row["audio_metadata"]["provenance"]["frame_count"] == full_frames
    assert row["audio_metadata"]["span"]["end_frame"] == min(full_frames, 96000)


async def test_projection_rejects_full_audio_beyond_30_seconds_without_duration_field(evaluation_graph):
    g = evaluation_graph
    await seed_metadata(g)
    issued = await g.issue(EvaluationModality.AUDIO_PREVIEW)
    projection = await selected_read(g, issued.actor, [])
    row = projection.rows[0]
    row.update(projection_audio_payload(480000, row["audio_metadata"]))
    EvaluationProjection.__post_init__(projection)
    assert "duration_s" not in row and "transcript_metadata" not in row
    # Keep the same valid six-second WAV/span and mutate only the full-input
    # counter. Preview agreement cannot admit an impossible >30-second input.
    row["audio_metadata"]["provenance"]["frame_count"] = 480001
    with pytest.raises(ValueError, match="invalid_evaluation_payload"):
        EvaluationProjection.__post_init__(projection)


async def test_projection_keeps_text_input_rate_and_window_independent_of_audio_bounds(evaluation_graph):
    g = evaluation_graph
    await seed_metadata(g)
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    projection = await selected_read(g, issued.actor, [])
    row = projection.rows[0]
    metadata = row["transcript_metadata"]
    metadata["provenance"]["sample_rate_hz"] = 8000
    metadata["provenance"]["frame_count"] = 480001
    metadata["span"] = {"start_frame": 0, "end_frame": 480001}
    EvaluationProjection.__post_init__(projection)
    assert row["transcript_metadata"]["provenance"]["sample_rate_hz"] == 8000
    assert row["transcript_metadata"]["span"]["end_frame"] == 480001
    assert "audio_metadata" not in row and "audio_base64" not in row


@pytest.mark.parametrize("field", ["sources_json", "expires_at"])
async def test_prepare_grant_uses_the_same_bounded_source_read(evaluation_graph, field):
    g = evaluation_graph
    selection = EvaluationGrantSelection(g.identity, frozenset({EvaluationModality.AUDIO_PREVIEW}), g.clock.now + 10)
    value = PRIVATE_MARKER * 9000 if field == "sources_json" else PRIVATE_MARKER * 1000
    await native.sql(g, f"UPDATE voice_fingerprints SET {field}=? WHERE id=?", (value, g.identity))
    await no_selected_row_materialized(g, g.voices.evaluation.prepare_grant,
        (g.root, (selection,), g.clock.now + 20), "sources_json")


async def test_known_metadata_does_not_resurrect_a_retired_original_login(evaluation_graph, monkeypatch):
    import src.users.voice_evaluation as module
    g = evaluation_graph
    await seed_metadata(g)
    issued = await g.issue(EvaluationModality.AUDIO_PREVIEW)
    entered, release = threading.Event(), threading.Event()
    original = module.b64encode
    def held(value):
        entered.set()
        assert release.wait(3)
        return original(value)
    monkeypatch.setattr(module, "b64encode", held)
    pending = asyncio.create_task(selected_read(g, issued.actor, []))
    try:
        async with asyncio.timeout(3):
            while not entered.is_set():
                await asyncio.sleep(.001)
        await g.authority.logout(issued.session.lease)
        release.set()
        with pytest.raises(PermissionError):
            await pending
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
