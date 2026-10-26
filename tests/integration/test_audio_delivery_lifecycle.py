"""Real zone/player routing with synthetic PCM, driver drain and cancellation."""

from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.audio.zone_manager import Zone, ZoneManager, ZoneState
from src.tts import piper_tts
from src.tts.piper_tts import AudioPlaybackError, PiperTTS, StreamingAudioPlayer
from tests.mocks.audio_output import AudioBackend


def zones():
    amplifier = MagicMock()
    amplifier.audio_output_device = 7
    manager = ZoneManager([Zone("zone_origin", "Origin", 0, 1), Zone("zone_other", "Other", 1, 2)], ma1260_controller=amplifier)
    manager.set_active_zone("zone_other")
    return manager, amplifier


def test_zone_stream_retains_every_sample_and_originating_route(monkeypatch):
    backend = AudioBackend()
    monkeypatch.setattr(piper_tts, "sd", backend)
    manager, amplifier = zones()
    samples = np.linspace(-1, 1, 24000, dtype=np.float32)
    handles = []
    assert manager.play_stream_to_zone("zone_origin", iter([samples]), sample_rate=24000, on_player=handles.append)
    np.testing.assert_array_equal(backend.heard()[:, 0], samples)
    assert len(handles) == 1 and handles[0].get_stats()["state"] == "completed"
    assert backend.outputs[0].device == 7
    amplifier.select_zone.assert_called_once_with(1)
    amplifier.set_volume.assert_called_once_with(1, 50)
    assert manager.get_zone("zone_origin").state is ZoneState.IDLE
    assert manager.get_zone("zone_other").state is ZoneState.IDLE


def test_zone_output_failure_propagates_without_success_or_automatic_retry(monkeypatch):
    backend = AudioBackend(stall=True)
    monkeypatch.setattr(piper_tts, "sd", backend)
    manager, _ = zones()
    closed = threading.Event()

    def source():
        try:
            yield np.ones(24000, dtype=np.float32)
        finally:
            closed.set()

    with pytest.raises(AudioPlaybackError):
        manager.play_stream_to_zone("zone_origin", source(), stall_timeout_s=0.1)
    assert closed.is_set()
    assert len(backend.outputs) == 1 and backend.outputs[0].closed
    assert manager.get_zone("zone_origin").state is ZoneState.IDLE


def test_cancel_during_registration_prevents_device_start(monkeypatch):
    backend = AudioBackend()
    monkeypatch.setattr(piper_tts, "sd", backend)
    manager, _ = zones()
    cancellation = threading.Event()

    def cancel(player):
        cancellation.set()
        player.stop()

    assert manager.play_stream_to_zone("zone_origin", iter([np.ones(24000, dtype=np.float32)]), on_player=cancel, cancel_event=cancellation) is False
    assert backend.outputs == []
    assert manager.get_zone("zone_origin").state is ZoneState.IDLE


def test_missing_zone_cannot_fall_back_to_the_current_zone(monkeypatch):
    backend = AudioBackend()
    monkeypatch.setattr(piper_tts, "sd", backend)
    manager, amplifier = zones()
    with pytest.raises(ValueError, match="zone"):
        manager.play_stream_to_zone("zone_missing", iter([np.ones(24000, dtype=np.float32)]))
    assert backend.outputs == []
    amplifier.select_zone.assert_not_called()


def test_piper_stream_exposes_the_same_player_that_owns_pcm(monkeypatch):
    backend = AudioBackend()
    monkeypatch.setattr(piper_tts, "sd", backend)
    engine = PiperTTS("synthetic.onnx")
    engine.sample_rate = 24000
    samples = np.arange(24000, dtype=np.float32) / 24000
    monkeypatch.setattr(engine, "synthesize_stream", lambda text: iter([samples]))
    handles = []
    assert engine.speak_stream("Synthetic text", on_player=handles.append)
    assert handles[0].get_stats()["total_samples"] == 24000
    np.testing.assert_array_equal(backend.heard()[:, 0], samples)


def test_late_chunk_from_cancelled_producer_cannot_restart_or_contaminate_output(monkeypatch):
    backend = AudioBackend()
    monkeypatch.setattr(piper_tts, "sd", backend)
    entered, release = threading.Event(), threading.Event()
    player = StreamingAudioPlayer(sample_rate=24000)

    def delayed():
        yield np.ones(10, dtype=np.float32)
        entered.set()
        assert release.wait(timeout=1)
        yield np.full(24000, 0.5, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(player.play_stream, delayed())
        try:
            assert entered.wait(1)
            player.stop()
            with pytest.raises(RuntimeError, match="producer"):
                player.start()
        finally:
            release.set()
        assert future.result(timeout=1) is False
    assert backend.outputs == []
    expected = np.full(24000, -0.5, dtype=np.float32)
    assert player.play_stream(iter([expected]))
    np.testing.assert_array_equal(backend.heard()[:, 0], expected)
