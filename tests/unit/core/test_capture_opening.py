"""Input-opening metadata cannot infer missing chunks or PCM-zero anchors."""

from collections import deque
from dataclasses import replace

import pytest

from src.core.capture_opening import (
    CaptureOpening, capture_opening_context, coherent_opening, current_capture_opening,
)
from src.evaluation.provenance import CaptureLane, CaptureSource, DSPCertainty, DSPObservation


def opening(**changes):
    values = dict(source=CaptureSource.LIVE_INPUT, source_ref='synthetic-opening',
        stream_generation=3, sample_rate_hz=16000, channels=2)
    values.update(changes)
    return CaptureOpening(**values)


@pytest.mark.parametrize('changes', [
    {'source': 'live_input'}, {'source_ref': ''}, {'source_ref': '/dev/private'},
    {'source_ref': 'x' * 65}, {'stream_generation': True}, {'stream_generation': -1},
    {'sample_rate_hz': 16000.0}, {'sample_rate_hz': 0}, {'channels': True},
    {'channels': 0}, {'channels': 65}, {'dsp': {}},
])
def test_opening_uses_closed_observed_format_fields(changes):
    with pytest.raises(ValueError):
        opening(**changes)


def test_projection_keeps_selected_pcm_and_independent_anchor_identity():
    source = opening()
    common = dict(lane=CaptureLane.COMMAND, segment_generation=12,
                  selected_channel=1, frame_count=16000)
    audio = source.project(**common, input_ref='cropped-12')
    text = source.project(**common, input_ref='stt-window-12')
    assert audio.source_ref != text.source_ref
    assert audio.frame_count == text.frame_count == 16000
    assert audio.selected_channel == text.selected_channel == 1
    assert audio.stream_generation == 3 and audio.segment_generation == 12
    assert audio.dsp.certainty is DSPCertainty.UNKNOWN and not audio.comparable


@pytest.mark.parametrize('input_ref', [None, True, '', '../window', 'x' * 63, 'b:c'])
def test_projection_does_not_coerce_or_guess_a_pcm_anchor(input_ref):
    with pytest.raises(ValueError):
        opening().project(lane=CaptureLane.AMBIENT, segment_generation=1,
            selected_channel=0, frame_count=16000, input_ref=input_ref)


def test_opening_separator_cannot_alias_another_pcm_input():
    first = opening(source_ref='a:b').project(lane=CaptureLane.COMMAND,
        segment_generation=1, selected_channel=0, frame_count=16000, input_ref='c')
    second = opening(source_ref='a').project(lane=CaptureLane.COMMAND,
        segment_generation=1, selected_channel=0, frame_count=16000, input_ref='b-c')
    assert first.source_ref == 'a:b:c'
    assert second.source_ref == 'a:b-c'
    assert first.source_ref != second.source_ref
    with pytest.raises(ValueError, match='invalid_capture_input_reference'):
        opening(source_ref='a').project(lane=CaptureLane.COMMAND,
            segment_generation=1, selected_channel=0, frame_count=16000, input_ref='b:c')


def test_opening_and_projection_detach_dsp_observation_values():
    intended = DSPObservation(DSPCertainty.DECLARED, 'before')
    source = opening(dsp=intended)
    object.__setattr__(intended, 'declared_revision', 'after')
    selected = source.project(lane=CaptureLane.AMBIENT, segment_generation=1,
        selected_channel=0, frame_count=16000, input_ref='segment-1')
    object.__setattr__(source.dsp, 'declared_revision', 'later')
    assert selected.dsp.declared_revision == 'before' and not selected.comparable


def test_unknown_input_is_not_promoted_by_a_later_known_packet():
    source = opening()
    assert coherent_opening([]) == (None, False)
    assert coherent_opening([None]) == (None, False)
    assert coherent_opening([None, source]) == (None, False)
    assert coherent_opening([source, None]) == (None, False)
    assert coherent_opening([source, replace(source)]) == (source, False)
    ring = deque([None, source], maxlen=2)
    assert coherent_opening(ring) == (None, False)
    ring.append(source)
    assert coherent_opening(ring) == (source, False)


@pytest.mark.parametrize('changes', [
    {'source_ref': 'new-opening'}, {'stream_generation': 4}, {'channels': 1},
    {'sample_rate_hz': 48000}, {'dsp': DSPObservation(DSPCertainty.DECLARED, 'new-dsp')},
])
def test_known_conflict_is_reported_even_with_unknown_contributions(changes):
    first, second = opening(), opening(**changes)
    assert coherent_opening([first, second]) == (None, True)
    assert coherent_opening([first, None, second]) == (None, True)


def test_packet_context_restores_unknown_and_prior_value_after_exception():
    source = opening()
    assert current_capture_opening() is None
    with capture_opening_context(source):
        assert current_capture_opening() is source
        with capture_opening_context(None):
            assert current_capture_opening() is None
        with pytest.raises(RuntimeError):
            with capture_opening_context(opening(source_ref='other')):
                raise RuntimeError('synthetic failure')
        assert current_capture_opening() is source
    assert current_capture_opening() is None
