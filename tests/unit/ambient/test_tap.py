"""Tests: MultiChannelTap — ring buffer del audio callback al ambient path."""
import numpy as np

from src.ambient.tap import MultiChannelTap


def _chunk(val: float = 0.1) -> np.ndarray:
    return np.full((1280, 6), val, dtype=np.float32)


def test_push_drain_roundtrip():
    tap = MultiChannelTap(maxlen_chunks=10)
    tap.register_room("escritorio")
    tap.push("escritorio", _chunk(0.1), ts=100.0)
    tap.push("escritorio", _chunk(0.2), ts=100.08, tts_active=True)

    items = tap.drain("escritorio")
    assert len(items) == 2
    ts0, chunk0, tts0 = items[0]
    ts1, chunk1, tts1 = items[1]
    assert ts0 == 100.0 and tts0 is False
    assert ts1 == 100.08 and tts1 is True
    assert chunk1[0, 0] == np.float32(0.2)
    # drain vacía la cola
    assert tap.drain("escritorio") == []


def test_maxlen_discards_oldest_fifo():
    tap = MultiChannelTap(maxlen_chunks=3)
    tap.register_room("escritorio")
    for i in range(5):
        tap.push("escritorio", _chunk(float(i)), ts=float(i))
    items = tap.drain("escritorio")
    assert len(items) == 3
    assert [ts for ts, _, _ in items] == [2.0, 3.0, 4.0]


def test_push_unregistered_room_is_noop():
    tap = MultiChannelTap(maxlen_chunks=3)
    tap.push("living", _chunk(), ts=1.0)  # no debe lanzar
    assert tap.drain("living") == []


def test_rooms_are_independent():
    tap = MultiChannelTap(maxlen_chunks=3)
    tap.register_room("escritorio")
    tap.register_room("living")
    tap.push("escritorio", _chunk(), ts=1.0)
    assert len(tap.drain("escritorio")) == 1
    assert tap.drain("living") == []
