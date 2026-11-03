"""Private proposal: coherent opening metadata follows every retained PCM chunk.

Imports target production modules after root integrates the staged copies. No
model, device or real audio is needed; the predictor and PCM are synthetic.
"""

import numpy as np
import pytest

from src.ambient.room_producer import RoomProducer
from src.ambient.segmenter import UtteranceSegmenter
from src.ambient.tap import TapPacket
from src.ambient.types import RawSegment
from src.core.capture_opening import CaptureOpening
from src.evaluation.provenance import CaptureSource, DSPCertainty, DSPObservation


SR = 16_000
CHUNK = 1280


def opening(*, generation=7, channels=2, **changes):
    values = dict(source=CaptureSource.LIVE_INPUT, source_ref="synthetic-opening",
                  stream_generation=generation, sample_rate_hz=SR, channels=channels,
                  dsp=DSPObservation(DSPCertainty.DECLARED, "synthetic-settings"))
    values.update(changes)
    return CaptureOpening(**values)


def chunk(value=0., channels=2):
    return np.full((CHUNK, channels), value, dtype=np.float32)


class Predictor:
    def __init__(self):
        self.events = []
        self.fail_reset = False

    def __call__(self, mono):
        value = float(mono[0])
        self.events.append(("predict", value))
        return .9 if value > .5 else 0.

    def reset(self):
        self.events.append(("reset",))
        if self.fail_reset:
            raise RuntimeError("synthetic reset failure")


def segmenter(*, preroll_ms=80, **changes):
    predictor = Predictor()
    values = dict(vad_predict=predictor, sample_rate=SR, vad_col=0,
                  speech_threshold=.5, close_silence_ms=80, preroll_ms=preroll_ms,
                  max_segment_s=30., min_speech_ms=80)
    values.update(changes)
    return UtteranceSegmenter(**values), predictor


def feed(seg, index, value, descriptor=None, *, channels=2, tts=False):
    return seg.feed(1. + index * .08, chunk(value, channels), tts, opening=descriptor)


def test_known_opening_covers_preroll_speech_and_tail_without_changing_pcm():
    seg, predictor = segmenter()
    known = opening()
    assert feed(seg, 0, .1, known) == []
    assert feed(seg, 1, .9, known, tts=True) == []
    [result] = feed(seg, 2, .2, known)
    assert result.input_opening == known and result.input_opening is not known
    assert result.segment_generation == 1
    assert result.audio.shape == (CHUNK * 3, 2)
    assert np.all(result.audio[:CHUNK] == np.float32(.1))
    assert np.all(result.audio[CHUNK:2 * CHUNK] == np.float32(.9))
    assert np.all(result.audio[2 * CHUNK:] == np.float32(.2))
    assert result.during_tts and result.complete
    assert predictor.events[-1] == ("reset",)


def test_unknown_preroll_downgrades_the_whole_emitted_segment():
    seg, _ = segmenter()
    known = opening()
    feed(seg, 0, 0., None)
    feed(seg, 1, .9, known)
    [result] = feed(seg, 2, 0., known)
    assert result.input_opening is None
    assert result.audio.shape[0] == CHUNK * 3


def test_unknown_descriptor_restores_only_after_its_pcm_leaves_preroll():
    seg, _ = segmenter(preroll_ms=80)
    known = opening()
    feed(seg, 0, .1, None)
    feed(seg, 1, .2, known)  # evicts the only unknown contribution
    feed(seg, 2, .9, known)
    [result] = feed(seg, 3, 0., known)
    assert result.input_opening == known
    assert np.all(result.audio[:CHUNK] == np.float32(.2))
    assert not np.any(result.audio == np.float32(.1))


def test_unknown_in_old_speech_buffer_is_not_replaced_by_many_new_known_chunks():
    seg, _ = segmenter(preroll_ms=80)
    known = opening()
    feed(seg, 0, .9, known)
    feed(seg, 1, .9, None)
    for index in range(2, 8):
        assert feed(seg, index, .9, known) == []
    [result] = feed(seg, 8, 0., known)
    assert result.input_opening is None
    assert result.audio.shape[0] == CHUNK * 9


def test_known_conflict_resets_before_predicting_and_preserves_current_packet():
    seg, predictor = segmenter()
    first, second = opening(), opening(generation=8)
    feed(seg, 0, .9, first)
    feed(seg, 1, .7, None)
    assert feed(seg, 2, .8, second) == []
    assert predictor.events[-2][0] == "reset"
    assert predictor.events[-1] == ("predict", float(np.float32(.8)))
    [result] = feed(seg, 3, 0., second)
    assert result.input_opening == second
    assert result.audio.shape[0] == CHUNK * 2
    assert np.all(result.audio[:CHUNK] == np.float32(.8))
    assert result.t0 == pytest.approx(1.16)


def test_recurrent_known_cohort_conflict_is_detected_after_its_pcm_was_evicted():
    seg, predictor = segmenter(preroll_ms=80)
    first, second = opening(), opening(generation=8)
    feed(seg, 0, .1, first)
    feed(seg, 1, .2, None)  # old known PCM is gone, but VAD state still saw it
    feed(seg, 2, .9, second)
    assert predictor.events[-2] == ("reset",)
    [result] = feed(seg, 3, 0., second)
    assert result.input_opening == second
    assert result.audio.shape[0] == CHUNK * 2  # unknown preroll was reset too


def test_known_channel_layout_change_never_concatenates_incompatible_pcm():
    seg, predictor = segmenter()
    feed(seg, 0, .9, opening(channels=2), channels=2)
    second = opening(generation=8, channels=3)
    feed(seg, 1, .8, second, channels=3)
    [result] = feed(seg, 2, 0., second, channels=3)
    assert result.audio.shape == (CHUNK * 2, 3)
    assert result.input_opening == second
    assert sum(event == ("reset",) for event in predictor.events) == 2


def test_descriptor_values_are_detached_from_a_producer_bypassing_frozen_types():
    seg, _ = segmenter()
    supplied, stable = opening(), opening()
    feed(seg, 0, .9, supplied)
    object.__setattr__(supplied, "stream_generation", 99)
    object.__setattr__(supplied.dsp, "declared_revision", "synthetic-after")
    [result] = feed(seg, 1, 0., stable)
    assert result.input_opening.stream_generation == 7
    assert result.input_opening.dsp.declared_revision == "synthetic-settings"


def test_segment_generation_is_monotonic_across_explicit_reset_and_discard():
    seg, _ = segmenter(preroll_ms=0)
    known = opening()
    feed(seg, 0, .9, known)
    [first] = feed(seg, 1, 0., known)
    feed(seg, 2, .1, None)
    seg.reset()
    feed(seg, 3, .9, known)
    [second] = feed(seg, 4, 0., known)
    assert second.segment_generation > first.segment_generation
    assert second.input_opening == known
    assert second.audio.shape[0] == CHUNK * 2


def test_failed_conflict_reset_admits_no_new_pcm_until_real_reset_succeeds():
    seg, predictor = segmenter()
    first, second = opening(), opening(generation=8)
    feed(seg, 0, .9, first)
    predictor.fail_reset = True
    with pytest.raises(RuntimeError, match="synthetic reset failure"):
        feed(seg, 1, .8, second)
    assert seg.needs_reset
    assert predictor.events[-1] == ("reset",)
    with pytest.raises(RuntimeError, match="ambient_segmenter_reset_pending"):
        feed(seg, 2, .8, second)
    predictor.fail_reset = False
    seg.reset()
    feed(seg, 3, .8, second)
    [result] = feed(seg, 4, 0., second)
    assert result.input_opening == second
    assert result.audio.shape[0] == CHUNK * 2


@pytest.mark.parametrize("descriptor", [opening(channels=1), opening(sample_rate_hz=8000), object()])
def test_invalid_or_mismatched_descriptor_is_rejected_before_vad(descriptor):
    seg, predictor = segmenter()
    with pytest.raises(ValueError):
        feed(seg, 0, .9, descriptor)
    assert predictor.events == []


def test_positional_raw_segment_construction_remains_unknown():
    result = RawSegment(1., 1.08, chunk(), False, .9, True, None, None)
    assert result.input_opening is result.segment_generation is None


@pytest.mark.asyncio
async def test_room_producer_legacy_tuples_do_not_add_kwargs_to_old_fake_segmenter():
    calls = []
    class Legacy:
        oldest_timestamp = None
        def feed(self, ts, chunk, tts_active=False):
            calls.append((ts, chunk, tts_active))
            return [RawSegment(ts, ts + .08, chunk)]
        def reset(self):
            pass
    producer = RoomProducer("synthetic-room", Legacy)
    try:
        audio = chunk(.9)
        [result] = producer._feed([(1., audio, False)])
        assert len(calls) == 1 and calls[0][0] == 1. and calls[0][2] is False
        assert calls[0][1] is audio
        assert result.input_opening is None
    finally:
        await producer.begin_close()


@pytest.mark.asyncio
async def test_room_producer_passes_each_packet_and_never_assigns_last_descriptor_to_outputs():
    first, second = opening(), opening(generation=8)
    calls = []
    class Stamped:
        oldest_timestamp = None
        def feed(self, ts, chunk, tts_active=False, opening=None):
            calls.append(opening)
            # These are deliberate segment-level results. The producer must not
            # replace an unknown/result-owned descriptor with the latest packet.
            descriptor = None if len(calls) == 1 else first
            return [RawSegment(ts, ts + .08, chunk, input_opening=descriptor, segment_generation=17)]
        def reset(self):
            pass
    producer = RoomProducer("synthetic-room", Stamped)
    try:
        packets = [TapPacket(1., chunk(.9), False, opening=first),
                   TapPacket(1.08, chunk(.8), False, opening=second)]
        results = producer._feed(packets)
        assert calls == [first, second]
        assert len(results) == 2
        assert results[0].input_opening is None
        assert results[1].input_opening == first
        assert results[1].segment_generation == 17
    finally:
        await producer.begin_close()
