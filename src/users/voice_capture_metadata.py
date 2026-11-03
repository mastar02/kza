"""Closed, bounded metadata codecs; no payload, clock or source-policy I/O."""
from __future__ import annotations

import json

from src.evaluation.provenance import (
    CaptureLane, CaptureProvenance, CaptureSource, DSPCertainty,
    DSPObservation, FrameInterval,
)

MAX_METADATA_BYTES = 4096
_ERROR = 'invalid_voice_capture_metadata'
_PROVENANCE_FIELDS = frozenset({
    'source', 'source_ref', 'lane', 'stream_generation', 'segment_generation',
    'source_channels', 'selected_channel', 'sample_rate_hz', 'frame_count', 'dsp',
})
_FAILURES = (ValueError, TypeError, AttributeError, KeyError, OverflowError, RecursionError)


def _fields(value, names) -> None:
    if type(value) is not dict or set(value) != names:
        raise ValueError(_ERROR)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(_ERROR)
        result[key] = value
    return result


def _constant(_value):
    raise ValueError(_ERROR)


def _pair(provenance, span) -> None:
    if provenance is None and span is None:
        return
    if type(provenance) is not CaptureProvenance or type(span) is not FrameInterval:
        raise ValueError(_ERROR)
    CaptureProvenance.__post_init__(provenance)
    FrameInterval.__post_init__(span)
    if span.end_frame > provenance.frame_count:
        raise ValueError(_ERROR)


def _provenance_document(value):
    if value is None:
        return None
    return {
        'source': value.source.value, 'source_ref': value.source_ref,
        'lane': value.lane.value, 'stream_generation': value.stream_generation,
        'segment_generation': value.segment_generation,
        'source_channels': value.source_channels, 'selected_channel': value.selected_channel,
        'sample_rate_hz': value.sample_rate_hz, 'frame_count': value.frame_count,
        'dsp': {'certainty': value.dsp.certainty.value,
                'declared_revision': value.dsp.declared_revision,
                'observed_revision': value.dsp.observed_revision},
    }


def _read_provenance(value):
    if value is None:
        return None
    _fields(value, _PROVENANCE_FIELDS)
    _fields(value['dsp'], {'certainty', 'declared_revision', 'observed_revision'})
    if any(type(value[key]) is not str for key in ('source', 'source_ref', 'lane')):
        raise ValueError(_ERROR)
    dsp = value['dsp']
    if type(dsp['certainty']) is not str:
        raise ValueError(_ERROR)
    return CaptureProvenance(
        source=CaptureSource(value['source']), source_ref=value['source_ref'],
        lane=CaptureLane(value['lane']), stream_generation=value['stream_generation'],
        segment_generation=value['segment_generation'], source_channels=value['source_channels'],
        selected_channel=value['selected_channel'], sample_rate_hz=value['sample_rate_hz'],
        frame_count=value['frame_count'],
        dsp=DSPObservation(DSPCertainty(dsp['certainty']), dsp['declared_revision'], dsp['observed_revision']),
    )


def _span_document(value):
    return None if value is None else {'start_frame': value.start_frame, 'end_frame': value.end_frame}


def _read_span(value):
    if value is None:
        return None
    _fields(value, {'start_frame', 'end_frame'})
    return FrameInterval(value['start_frame'], value['end_frame'])


def _encode(provenance, span, text_truncated, *, transcript):
    _pair(provenance, span)
    if transcript:
        if text_truncated is not None and type(text_truncated) is not bool:
            raise ValueError(_ERROR)
        if provenance is None and text_truncated is None:
            return None
    elif provenance is None:
        return None
    document = {'schema_version': 1, 'provenance': _provenance_document(provenance),
                'span': _span_document(span)}
    if transcript:
        document['text_truncated'] = text_truncated
    raw = json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    if len(raw.encode('utf-8')) > MAX_METADATA_BYTES:
        raise ValueError(_ERROR)
    return raw


def _decode(raw, *, transcript):
    if (type(raw) is not str or not raw or len(raw) > MAX_METADATA_BYTES
            or len(raw.encode('utf-8')) > MAX_METADATA_BYTES):
        raise ValueError(_ERROR)
    document = json.loads(raw, object_pairs_hook=_unique, parse_constant=_constant)
    expected = {'schema_version', 'provenance', 'span'}
    if transcript:
        expected.add('text_truncated')
    _fields(document, expected)
    if type(document['schema_version']) is not int or document['schema_version'] != 1:
        raise ValueError(_ERROR)
    provenance, span = _read_provenance(document['provenance']), _read_span(document['span'])
    _pair(provenance, span)
    if transcript:
        truncated = document['text_truncated']
        if truncated is not None and type(truncated) is not bool:
            raise ValueError(_ERROR)
        if provenance is None and truncated is None:
            raise ValueError(_ERROR)
        return provenance, span, truncated
    if provenance is None:
        raise ValueError(_ERROR)
    return provenance, span


def encode_audio_metadata(provenance: CaptureProvenance | None, span: FrameInterval | None) -> str | None:
    """SQL NULL represents an entirely unknown audio descriptor."""
    invalid = False
    try:
        return _encode(provenance, span, None, transcript=False)
    except _FAILURES:
        invalid = True
    finally:
        provenance = span = None
    if invalid:
        raise ValueError(_ERROR)


def decode_audio_metadata(raw: str | None) -> tuple[CaptureProvenance | None, FrameInterval | None]:
    """Decode an audio-only document; SQL NULL keeps its provenance unknown."""
    invalid = False
    try:
        if raw is None:
            return None, None
        return _decode(raw, transcript=False)
    except _FAILURES:
        invalid = True
    finally:
        raw = None
    if invalid:
        raise ValueError(_ERROR)


def encode_transcript_metadata(provenance: CaptureProvenance | None, span: FrameInterval | None,
                               text_truncated: bool | None) -> str | None:
    """The truncation flag may be known independently of PCM provenance."""
    invalid = False
    try:
        return _encode(provenance, span, text_truncated, transcript=True)
    except _FAILURES:
        invalid = True
    finally:
        provenance = span = text_truncated = None
    if invalid:
        raise ValueError(_ERROR)


def decode_transcript_metadata(raw: str | None) -> tuple[CaptureProvenance | None, FrameInterval | None, bool | None]:
    """Decode text-only provenance and a producer-known truncation flag."""
    invalid = False
    try:
        if raw is None:
            return None, None, None
        return _decode(raw, transcript=True)
    except _FAILURES:
        invalid = True
    finally:
        raw = None
    if invalid:
        raise ValueError(_ERROR)
