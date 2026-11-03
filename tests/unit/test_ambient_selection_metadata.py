"""Private proposed tests; synthetic inputs and engines only."""
from dataclasses import replace
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from src.ambient.ambient_stt import AmbientSTT, AmbientSTTResult
from src.ambient.transcriber import AmbientTranscriber
from src.ambient.types import RawSegment
from src.core.capture_opening import CaptureOpening
from src.evaluation.provenance import CaptureLane, CaptureSource, DSPObservation, FrameInterval
from src.stt.whisper_fast import STTResult


def opening(channels=2, **changes):
    values = dict(source=CaptureSource.LIVE_INPUT, source_ref='synthetic-input',
        stream_generation=7, sample_rate_hz=16000, channels=channels, dsp=DSPObservation())
    values.update(changes)
    return CaptureOpening(**values)


class Engine:
    def __init__(self, text='Synthetic full transcript'):
        self.text, self.selected_first, self.before_return = text, None, None

    def transcribe_with_confidence(self, mono):
        self.selected_first = float(mono[0])
        if self.before_return is not None:
            self.before_return()
        return STTResult(self.text, 1., avg_logprob=-.1)


@pytest.mark.asyncio
@pytest.mark.parametrize('known', [False, True])
async def test_native_selection_survives_configuration_change_without_retaining_pcm(known):
    engine = Engine()
    adapter = AmbientSTT(engine, asr_col=1)
    engine.before_return = lambda: setattr(adapter, 'asr_col', 0)
    audio = np.zeros((16001, 2), dtype=np.float32)
    audio[:, 1] = .5
    try:
        result = await adapter.transcribe(audio, opening=opening() if known else None,
                                         segment_generation=11)
        assert type(result) is AmbientSTTResult
        assert engine.selected_first == .5 and result.selected_channel == 1
        assert result.mono_from(audio)[0] == np.float32(.5)
        assert result.text == engine.text and result.avg_logprob == -.1
        assert not hasattr(result, '__dict__') and not hasattr(result, 'result')
        if known:
            assert result.provenance.selected_channel == 1
            assert result.provenance.frame_count == 16001
            assert result.provenance.segment_generation == 11
            assert result.span == FrameInterval(0, 16001)
        else:
            assert result.provenance is None and result.span is None
    finally:
        await adapter.begin_close()


@pytest.mark.asyncio
async def test_actual_missing_asr_column_falls_back_to_zero_and_gets_distinct_anchors():
    adapter = AmbientSTT(Engine(), asr_col=4)
    audio = np.zeros((16000, 1), dtype=np.float32)
    try:
        first = await adapter.transcribe(audio, opening=opening(1), segment_generation=1)
        second = await adapter.transcribe(audio, opening=opening(1), segment_generation=1)
        assert first.selected_channel == first.provenance.selected_channel == 0
        assert first.provenance.source_ref != second.provenance.source_ref
    finally:
        await adapter.begin_close()


@pytest.mark.asyncio
@pytest.mark.parametrize('source', [opening(1), opening(sample_rate_hz=8000)])
async def test_known_opening_mismatch_fails_before_engine(source):
    engine = Engine()
    adapter = AmbientSTT(engine)
    try:
        with pytest.raises(ValueError):
            await adapter.transcribe(np.zeros((16000, 2), dtype=np.float32),
                opening=source, segment_generation=1)
        assert engine.selected_first is None
    finally:
        await adapter.begin_close()


def test_frozen_selection_rejects_frame_channel_and_descriptor_rebinding():
    provenance = opening().project(lane=CaptureLane.AMBIENT, segment_generation=1,
        selected_channel=1, frame_count=16000, input_ref='a' * 32)
    result = AmbientSTTResult('Synthetic text', 1., None, None, None, 1, 2, 16000,
        provenance, FrameInterval(0, 16000))
    for changes in ({'selected_channel': 0}, {'frame_count': 16001}, {'source_channels': True},
                    {'span': FrameInterval(1, 16000)}, {'provenance': None}):
        with pytest.raises(ValueError):
            replace(result, **changes)


@pytest.mark.asyncio
async def test_mutable_native_result_and_nested_extras_are_not_retained():
    pcm = np.zeros(16000, dtype=np.float32)
    raw = STTResult('Synthetic original text', 1., no_speech_prob=.1,
                    avg_logprob=-.2, compression_ratio=1.)
    raw.extra = {'nested': [pcm]}
    pcm_ref, raw_ref = weakref.ref(pcm), weakref.ref(raw)

    class MutableEngine:
        result = raw

        def transcribe_with_confidence(self, audio):
            return self.result

    engine = MutableEngine()
    adapter = AmbientSTT(engine)
    try:
        result = await adapter.transcribe(np.ones(16000, dtype=np.float32))
        raw.text, raw.elapsed_ms, raw.no_speech_prob = 'Synthetic replaced text', 9., .9
        raw.avg_logprob, raw.compression_ratio = -.9, 9.
        assert (result.text, result.elapsed_ms, result.no_speech_prob,
                result.avg_logprob, result.compression_ratio) == ('Synthetic original text', 1., .1, -.2, 1.)
        engine.result = MutableEngine.result = None
        raw = pcm = None
        await adapter.begin_close()
        gc.collect()
        assert raw_ref() is None and pcm_ref() is None
        assert not hasattr(result, '__dict__') and not hasattr(result, 'result')
    finally:
        engine.result = MutableEngine.result = None
        raw = pcm = None
        await adapter.begin_close()


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [('text', object()), ('elapsed_ms', True),
    ('elapsed_ms', float('inf')), ('no_speech_prob', np.zeros(1, dtype=np.float32)),
    ('avg_logprob', float('nan')), ('compression_ratio', {'nested': 'invalid'})])
async def test_native_result_fields_are_closed_finite_scalars(field, value):
    raw = STTResult('Synthetic text', 1.)
    setattr(raw, field, value)

    class InvalidEngine:
        def transcribe_with_confidence(self, audio):
            return raw

    adapter = AmbientSTT(InvalidEngine())
    try:
        with pytest.raises(ValueError, match='invalid_ambient_stt_result'):
            await adapter.transcribe(np.ones(16000, dtype=np.float32))
    finally:
        await adapter.begin_close()


@pytest.mark.asyncio
async def test_transcriber_keeps_review_ch0_full_stt_span_and_fixed_wake_pcm():
    engine, seen = Engine('x' * 8193), {}
    adapter = AmbientSTT(engine, asr_col=1)
    engine.before_return = lambda: setattr(adapter, 'asr_col', 0)

    class Tagger:
        review_managed = True

        async def tag(self, mono, **kwargs):
            seen['tag_first'], seen['review'] = float(mono[0]), kwargs
            return 'unknown', 0.

    class Store:
        async def add(self, utterance):
            return 1

    class Wake:
        async def maybe_dispatch(self, room, text, source, speaker, **kwargs):
            seen['wake_text'], seen['wake'] = text, kwargs

    class Native:
        async def run(self, function, *args):
            return function(*args)

    tr = object.__new__(AmbientTranscriber)
    tr._stt, tr._tagger, tr._archiver, tr._quality_fn = adapter, Tagger(), None, None
    tr._native, tr._store, tr._textual_wake = Native(), Store(), Wake()
    tr._classifier = SimpleNamespace(classify=lambda **kwargs: 'unknown')
    tr._last_tv, tr._require_live = {}, lambda source: None
    tr._direction = lambda audio, source: None
    audio = np.zeros((16001, 2), dtype=np.float32)
    audio[:, 0], audio[:, 1] = .25, .5
    segment = RawSegment(1., 1. + len(audio) / 16000, audio,
        input_opening=opening(), segment_generation=19)
    try:
        await tr._process_segment('synthetic-room', segment, None)
        review, wake = seen['review'], seen['wake']
        assert seen['tag_first'] == .25 and len(review['transcript']) == 8192
        assert review['transcript_truncated'] is True
        assert review['audio_provenance'].selected_channel == 0
        assert review['transcript_provenance'].selected_channel == 1
        assert review['transcript_span'] == FrameInterval(0, 16001)
        assert review['audio_provenance'].source_ref != review['transcript_provenance'].source_ref
        assert len(seen['wake_text']) == 8193 and wake['audio'][0] == np.float32(.5)
        assert wake['audio_provenance'] == wake['transcript_provenance'] == review['transcript_provenance']
        assert wake['transcript_truncated'] is False
    finally:
        seen.clear()
        await adapter.begin_close()
