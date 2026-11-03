"""Synthetic packet lineage through command and ambient ingress queues."""

from dataclasses import replace

import numpy as np
import pytest

from src.ambient.tap import MultiChannelTap
from src.core.capture_opening import (
    CaptureOpening,
    capture_opening_context,
    current_capture_opening,
)
from src.evaluation.provenance import CaptureSource
from src.pipeline.capture_room import CaptureRoom


def _opening() -> CaptureOpening:
    return CaptureOpening(CaptureSource.LIVE_INPUT, 'synthetic', 1, 16000, 2)


@pytest.mark.parametrize('fail', [False, True])
def test_command_packet_context_is_individual_and_restored(fail):
    first = _opening()
    second = replace(first, source_ref='next-opening', stream_generation=2)
    parent = replace(first, source_ref='outer')
    observed = []
    polled = []
    cleared = []

    def process(audio, frames, timestamp, status):
        observed.append((current_capture_opening(), float(audio[0, 0]), frames))
        if fail and len(observed) == 2:
            raise RuntimeError('synthetic native failure')

    def poll():
        polled.append(current_capture_opening())
        return None

    room = CaptureRoom('synthetic', process=process, poll=poll,
                       clear=lambda: cleared.append(True), listening=lambda: False)
    original = np.full((8, 2), 1, dtype=np.float32)
    room.push(1.0, original, 8, opening=first)
    original.fill(9)
    room.push(2.0, original, 8)
    room.push(3.0, original, 8, opening=second)
    packets = list(room._queue)
    assert packets[0].opening is first
    assert not np.shares_memory(original, packets[0].audio)
    with capture_opening_context(parent):
        assert room._advance(packets) is None
        assert current_capture_opening() is parent
    assert current_capture_opening() is None
    expected = [(first, 1.0, 8), (None, 9.0, 8), (second, 9.0, 8)]
    assert observed == (expected[:2] if fail else expected)
    assert cleared == ([True] if fail else [])
    assert polled == ([] if fail else [parent])


def test_ambient_owned_drain_retains_each_opening_without_another_pcm_copy():
    tap = MultiChannelTap(maxlen_chunks=4)
    tap.register_room('synthetic')
    first = _opening()
    second = replace(first, stream_generation=2)
    audio = np.zeros((8, 2), dtype=np.float32)
    tap.push('synthetic', audio, ts=1.0, opening=first)
    tap.push('synthetic', audio, ts=2.0)
    tap.push('synthetic', audio, ts=3.0, opening=second)
    packets = tap.drain_owned('synthetic')
    assert [packet.opening for packet in packets] == [first, None, second]
    assert all(packet.audio is audio for packet in packets)
    assert tap.drain_owned('synthetic') == []
