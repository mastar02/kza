"""Sample conservation, bounded producer pressure and output lifecycle."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import pytest

from src.tts import piper_tts
from src.tts.piper_tts import AudioPlaybackError, StreamingAudioPlayer
from tests.mocks.audio_output import AudioBackend


@pytest.mark.parametrize("sizes", [(24000,), (1,), (), (1, 63, 499, 8, 24000, 129, 60000)])
@pytest.mark.parametrize("channels", [1, 2])
def test_every_frame_survives_large_chunks_and_partial_tail(monkeypatch, sizes, channels):
    backend = AudioBackend()
    monkeypatch.setattr(piper_tts, "sd", backend)
    length = sum(sizes)
    expected = np.arange(length * channels, dtype=np.float32).reshape(length, channels)
    splits = np.cumsum(sizes)[:-1]
    source = np.split(expected, splits) if sizes else []
    player = StreamingAudioPlayer(sample_rate=24000, channels=channels)
    assert player.play_stream(iter(source)) is True
    if length:
        np.testing.assert_array_equal(backend.heard(), expected)
        assert all(output.error is None and output.closed for output in backend.outputs)
    else:
        assert backend.outputs == []
    stats = player.get_stats()
    assert stats["total_samples"] == stats["played_samples"] == length
    assert stats["queued_samples"] == 0 and stats["state"] == "completed"
    assert stats["max_buffered_samples"] <= player.buffer_samples


def test_callback_larger_than_ring_cannot_drop_queued_samples(monkeypatch):
    backend = AudioBackend(frame_sizes=(1000, 17, 500))
    monkeypatch.setattr(piper_tts, "sd", backend)
    expected = np.linspace(-1, 1, 24000, dtype=np.float32)
    player = StreamingAudioPlayer(sample_rate=24000, buffer_ms=5, prebuffer_ms=1)
    assert player.play_stream(iter([expected])) is True
    np.testing.assert_array_equal(backend.heard()[:, 0], expected)
    assert player.get_stats()["software_underruns"] > 0
    assert player.get_stats()["max_buffered_samples"] <= 120


def test_finish_waits_for_device_tail_instead_of_closing_after_last_callback(monkeypatch):
    backend = AudioBackend(hold_tail=True)
    monkeypatch.setattr(piper_tts, "sd", backend)
    player = StreamingAudioPlayer(sample_rate=24000)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(player.play_stream, iter([np.ones(241, dtype=np.float32)]))
        assert backend.created.wait(1)
        output = backend.outputs[0]
        assert output.draining.wait(1)
        assert not future.done() and not output.closed
        backend.release_tail.set()
        assert future.result(timeout=1) is True
    assert output.closed and output.error is None


def test_full_ring_blocks_the_producer_until_cancel_without_consuming_the_next_chunk(monkeypatch):
    backend = AudioBackend(stall=True)
    monkeypatch.setattr(piper_tts, "sd", backend)
    player = StreamingAudioPlayer(sample_rate=24000)
    closed = threading.Event()
    consumed = []

    def source():
        try:
            consumed.append(1)
            yield np.ones(24000, dtype=np.float32)
            consumed.append(2)
            yield np.ones(24000, dtype=np.float32)
        finally:
            closed.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(player.play_stream, source())
        assert backend.created.wait(1)
        assert not future.done() and consumed == [1]
        assert player.get_stats()["queued_samples"] == player.buffer_samples
        started = time.perf_counter()
        player.stop()
        assert future.result(timeout=1) is False
        assert time.perf_counter() - started <= 0.2
    assert closed.is_set() and backend.outputs[0].closed
    assert player.get_stats()["state"] == "cancelled"


def test_cancel_active_output_stops_device_and_producer_within_budget(monkeypatch):
    backend = AudioBackend()
    backend.callback_pause_s = 0.003
    monkeypatch.setattr(piper_tts, "sd", backend)
    player = StreamingAudioPlayer(sample_rate=24000)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(player.play_stream, iter([np.ones(240000, dtype=np.float32)]))
        assert backend.created.wait(1)
        assert backend.outputs[0].delivered.wait(1)
        started = time.perf_counter()
        player.stop()
        assert future.result(timeout=1) is False
        assert time.perf_counter() - started <= 0.2
    assert 0 < player.get_stats()["played_samples"] < 240000
    assert backend.outputs[0].closed


@pytest.mark.parametrize("failure", ["buffer_stall", "drain_stall", "driver_start", "driver_dies"])
def test_output_failure_raises_and_closes_instead_of_reporting_success(monkeypatch, failure):
    backend = AudioBackend(stall=failure == "buffer_stall", hold_tail=failure == "drain_stall",
        fail_start=failure == "driver_start", fail_after=1 if failure == "driver_dies" else None)
    monkeypatch.setattr(piper_tts, "sd", backend)
    player = StreamingAudioPlayer(sample_rate=24000, stall_timeout_s=0.1, drain_timeout_s=0.1)
    samples = 100 if failure == "drain_stall" else 24000
    with pytest.raises((AudioPlaybackError, RuntimeError)):
        player.play_stream(iter([np.ones(samples, dtype=np.float32)]))
    assert player.get_stats()["state"] == "failed"
    assert all(output.closed for output in backend.outputs)


def test_cancelled_token_cannot_start_late_audio(monkeypatch):
    backend = AudioBackend()
    monkeypatch.setattr(piper_tts, "sd", backend)
    cancelled = threading.Event()
    cancelled.set()
    player = StreamingAudioPlayer(cancel_event=cancelled)
    assert player.play_stream(iter([np.ones(24000, dtype=np.float32)])) is False
    assert backend.outputs == [] and player.get_stats()["state"] == "cancelled"


def test_token_cancel_during_device_drain_never_becomes_completion(monkeypatch):
    backend = AudioBackend(hold_tail=True)
    monkeypatch.setattr(piper_tts, "sd", backend)
    cancelled = threading.Event()
    player = StreamingAudioPlayer(cancel_event=cancelled)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(player.play_stream, iter([np.ones(100, dtype=np.float32)]))
        assert backend.created.wait(1)
        assert backend.outputs[0].draining.wait(1)
        started = time.perf_counter()
        cancelled.set()
        assert future.result(timeout=1) is False
        assert time.perf_counter() - started <= 0.2
    assert player.get_stats()["state"] == "cancelled" and backend.outputs[0].closed


def test_invalid_none_chunk_is_a_failure_instead_of_silent_end_of_response(monkeypatch):
    backend = AudioBackend()
    monkeypatch.setattr(piper_tts, "sd", backend)
    player = StreamingAudioPlayer()
    with pytest.raises(ValueError):
        player.play_stream(iter([None, np.ones(24000, dtype=np.float32)]))
    assert backend.outputs == [] and player.get_stats()["state"] == "failed"


@pytest.mark.parametrize("parameters", [{"buffer_ms": 0}, {"prebuffer_ms": 151}, {"channels": 0}, {"stall_timeout_s": float("inf")}, {"drain_timeout_s": 0}])
def test_invalid_bounds_cannot_create_a_nonprogressing_ring(parameters):
    with pytest.raises(ValueError):
        StreamingAudioPlayer(**parameters)
