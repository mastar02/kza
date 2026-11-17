"""Post-wake listening rule (V1, voice-diag 2026-11-17).

Replays the diagnosis scenarios through the production audio processor and
poll: Cocina-like quiet speech (RMS 0.003) with and without a pause after the
wake, loud near-field speech with 0.25/0.6 s pauses and without, silence after
the wake (drop without STT), follow-up start in a quiet room, plus latency
bounds of every close.
"""

import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import numpy as np
import pytest
import yaml

from tests.mocks.responses import response_handler_mock
from src.nlu.command_gate import CommandAcceptanceGate
from src.pipeline.multi_room_audio_loop import CHUNK_SIZE, MultiRoomAudioLoop, RoomStream
from src.pipeline.request_router import RequestRouter
from src.pipeline.post_wake_endpoint import (
    CaptureTrigger, EndpointDecision, EndpointState, NoiseFloorTracker, PostWakeConfig, classify_frame,
)

SR = 16000
CHUNK_S = CHUNK_SIZE / SR
CONFIG = PostWakeConfig()
DISPATCH_AT_MAX = replace(CONFIG, dispatch_at_max_utterance=True)
END_SILENCE_S = CONFIG.end_silence_ms / 1000
# The close lands on the first whole frame of silence at or past end_silence.
END_CLOSE_S = math.ceil(CONFIG.end_silence_ms / 1000 / CHUNK_S - 1e-9) * CHUNK_S
# Legacy endpoint: close 0.24 s after the end of speech (review probe P2).
LEGACY_CLOSE_S = 0.24


class _Detector:
    """openwakeword-like detector: fires once on a chosen chunk."""

    peek_pending_text = None
    pop_pending_command_audio = None
    pop_pending_text = None

    def __init__(self, fire_at: int | None):
        self.fire_at = fire_at
        self.calls = 0

    def load(self):
        pass

    def reset(self):
        pass

    def detect(self, chunk):
        index = self.calls
        self.calls += 1
        return ("nexa", 0.62) if index == self.fire_at else None


def _echo():
    m = MagicMock()
    m.is_safe_to_listen = True
    m.should_process_audio = MagicMock(return_value=(True, "ok"))
    m.is_human_voice = MagicMock(return_value=True)
    return m


def _follow_up(active=False):
    m = MagicMock()
    m.is_active = active
    return m


def _segment(rng, seconds, rms):
    n = int(round(seconds * SR))
    return (rng.standard_normal(n) * rms).astype(np.float32)


def _chunks(audio):
    usable = len(audio) - len(audio) % CHUNK_SIZE
    return [audio[i:i + CHUNK_SIZE] for i in range(0, usable, CHUNK_SIZE)]


def _loop(detector, post_wake=CONFIG, follow_up=None, **kwargs):
    rs = RoomStream(room_id="cocina", device_index=0, wake_detector=detector, echo_suppressor=_echo())
    loop = MultiRoomAudioLoop(room_streams={"cocina": rs}, follow_up=follow_up or _follow_up(),
                              wake_preroll_s=1.0, post_wake=post_wake, **kwargs)
    return loop, rs


def _run(*, floor, nexa_rms, pause_s, command_s, command_rms, tail_s=7.0, post_wake=CONFIG,
         seed=7, tail_rms=None, **kwargs):
    """Feed ambient, "Nexa", the wake, a pause, a command and a long tail.

    Returns:
        dict with the loop, room, the event (or None), how long after the
        wake the capture ended, and whether it was still listening.
    """
    rng = np.random.default_rng(seed)
    before = np.concatenate([_segment(rng, 3.04, floor), _segment(rng, 0.48, nexa_rms)])
    pre_chunks = _chunks(before)
    after = np.concatenate([
        _segment(rng, pause_s, floor if tail_rms is None else tail_rms),
        _segment(rng, command_s, command_rms), _segment(rng, tail_s, floor),
    ])
    detector = _Detector(fire_at=len(pre_chunks) - 1)
    loop, rs = _loop(detector, post_wake=post_wake, **kwargs)
    process = loop._make_audio_processor(rs)
    for chunk in pre_chunks:
        process(chunk[:, None], CHUNK_SIZE, None, None)
    assert rs.listening, "wake must open the capture"
    preroll = len(rs.audio_buffer)
    event = None
    closed_after = None
    for index, chunk in enumerate(_chunks(after), start=1):
        process(chunk[:, None], CHUNK_SIZE, None, None)
        event = loop._poll_command(rs)
        if event is not None or not rs.listening:
            closed_after = index * CHUNK_S
            break
    return {"loop": loop, "rs": rs, "event": event, "closed_after": closed_after, "preroll": preroll}


# ---------------------------------------------------------------- scenarios

@pytest.mark.parametrize("pause_s", [0.0, 0.24])
def test_cocina_like_quiet_speech_is_captured_whole(pause_s):
    command_end = pause_s + 1.2
    out = _run(floor=0.0012, nexa_rms=0.0035, pause_s=pause_s, command_s=1.2, command_rms=0.0032)
    event = out["event"]
    assert event is not None and event.capture_complete is True
    assert event.capture_trigger == "wake"
    # The whole command after the wake reached STT, plus the pre-roll.
    assert len(event.audio) >= out["preroll"] + int(command_end * SR)
    # Latency bound: closes end_silence after the command, within one frame.
    assert command_end + END_SILENCE_S - 1e-9 <= out["closed_after"] <= command_end + END_SILENCE_S + CHUNK_S + 1e-9


@pytest.mark.parametrize("pause_s", [0.24, 0.64])
def test_loud_near_field_with_pause_waits_for_the_command(pause_s):
    command_end = pause_s + 1.2
    out = _run(floor=0.003, nexa_rms=0.05, pause_s=pause_s, command_s=1.2, command_rms=0.03)
    event = out["event"]
    assert event is not None and event.capture_complete is True
    assert len(event.audio) - out["preroll"] >= int(command_end * SR)
    assert command_end + END_SILENCE_S - 1e-9 <= out["closed_after"] <= command_end + END_SILENCE_S + CHUNK_S + 1e-9


def test_loud_near_field_without_pause():
    out = _run(floor=0.003, nexa_rms=0.05, pause_s=0.0, command_s=1.04, command_rms=0.03)
    assert out["event"] is not None
    assert 1.04 + END_SILENCE_S - 1e-9 <= out["closed_after"] <= 1.04 + END_SILENCE_S + CHUNK_S + 1e-9


def test_pause_longer_than_end_silence_before_the_command_is_still_captured():
    """No speech after the wake yet: a 1.6 s pause cannot close the capture."""
    out = _run(floor=0.0012, nexa_rms=0.0035, pause_s=1.6, command_s=0.8, command_rms=0.0032)
    assert out["event"] is not None
    assert out["closed_after"] >= 1.6 + 0.8 + END_SILENCE_S - 1e-9


def test_silence_after_wake_is_dropped_without_stt():
    guard = MagicMock()
    guard.state_for.return_value = None
    stt = MagicMock()
    out = _run(floor=0.0012, nexa_rms=0.0035, pause_s=0.0, command_s=0.0, command_rms=0.0,
               ambient_guard=guard, stt=stt)
    assert out["event"] is None
    assert out["rs"].listening is False and out["rs"].audio_buffer == []
    # Dropped once no_speech_timeout_s of audio after the wake had no speech.
    assert CONFIG.no_speech_timeout_s - 1e-9 <= out["closed_after"] <= CONFIG.no_speech_timeout_s + CHUNK_S + 1e-9
    guard.on_capture_result.assert_called_once_with("cocina", "empty")
    stt.transcribe.assert_not_called()
    assert out["loop"].endpoint_counts == {"drop_no_speech": 1}


@pytest.mark.parametrize("tail_frames", [1, 2])
def test_wake_word_tail_does_not_delay_the_drop(tail_frames):
    """1-2 frames of the "Nexa" tail after the detection, then silence: drop at 2.5 s."""
    guard = MagicMock()
    guard.state_for.return_value = None
    out = _run(floor=0.0012, nexa_rms=0.0035, pause_s=tail_frames * CHUNK_S, command_s=0.0, command_rms=0.0,
               ambient_guard=guard, tail_s=7.0, seed=19, tail_rms=0.0035)
    assert out["event"] is None and out["rs"].listening is False
    assert CONFIG.no_speech_timeout_s - 1e-9 <= out["closed_after"] <= CONFIG.no_speech_timeout_s + CHUNK_S + 1e-9
    assert out["loop"].endpoint_counts == {"drop_no_speech": 1}
    guard.on_capture_result.assert_called_once_with("cocina", "empty")


def test_endpoint_state_drops_unconfirmed_speech_at_the_no_speech_timeout():
    for tail_frames in (1, 2):
        state = EndpointState(wake_index=0, floor=0.0012, next_frame=0)
        buffer = list(np.full(tail_frames * CHUNK_SIZE, 0.0035, dtype=np.float32))
        frames = math.ceil(CONFIG.no_speech_timeout_s / CHUNK_S - 1e-9)
        buffer += list(np.full((frames - tail_frames) * CHUNK_SIZE, 0.0012, dtype=np.float32))
        state.analyze(buffer[:-CHUNK_SIZE], CONFIG, None)
        assert state.decide(CONFIG, SR, False, 0.0) is EndpointDecision.CONTINUE
        state.analyze(buffer, CONFIG, None)
        assert state.speech_frames == tail_frames
        assert state.decide(CONFIG, SR, False, 0.0) is EndpointDecision.DROP_NO_SPEECH


def test_short_speech_waits_for_the_minimum_listen():
    out = _run(floor=0.003, nexa_rms=0.05, pause_s=0.0, command_s=0.32, command_rms=0.03)
    assert out["event"] is not None
    assert out["closed_after"] >= CONFIG.min_listen_s - 1e-9
    assert out["closed_after"] <= CONFIG.min_listen_s + CHUNK_S + 1e-9


def test_continuous_speech_is_capped_at_max_utterance_and_dispatched():
    """Opt-in dispatch at the cap (VC-Q2 pending; settings keep it off)."""
    out = _run(floor=0.003, nexa_rms=0.05, pause_s=0.0, command_s=8.0, command_rms=0.03,
               post_wake=DISPATCH_AT_MAX)
    event = out["event"]
    assert event is not None and event.capture_complete is True
    assert event.endpoint_code == "close_max_utterance"
    assert abs(out["closed_after"] - CONFIG.max_utterance_s) <= CHUNK_S + 1e-9
    assert out["loop"].endpoint_counts == {"close_max_utterance": 1}


def test_continuous_speech_at_the_cap_is_incomplete_by_default():
    out = _run(floor=0.003, nexa_rms=0.05, pause_s=0.0, command_s=8.0, command_rms=0.03)
    event = out["event"]
    assert event is not None and event.capture_complete is False
    assert event.endpoint_code == "close_max_utterance"
    assert abs(out["closed_after"] - CONFIG.max_utterance_s) <= CHUNK_S + 1e-9


def _false_wake_into_background_speech(post_wake=CONFIG):
    """A false wake followed by a call in the escritorio (rms 0.006, Silero 0.8)."""
    rng = np.random.default_rng(21)
    before = np.concatenate([_segment(rng, 3.04, 0.002), _segment(rng, 0.48, 0.006)])
    pre_chunks = _chunks(before)
    loop, rs = _loop(_Detector(fire_at=len(pre_chunks) - 1), post_wake=post_wake)
    loop._wake_vad_predictors["cocina"] = lambda _audio: 0.8
    process = loop._make_audio_processor(rs)
    for chunk in pre_chunks:
        process(chunk[:, None], CHUNK_SIZE, None, None)
    assert rs.listening
    for index, chunk in enumerate(_chunks(_segment(rng, 8.0, 0.006)), start=1):
        process(chunk[:, None], CHUNK_SIZE, None, None)
        event = loop._poll_command(rs)
        if event is not None:
            return event, index * CHUNK_S
    return None, None


def _router(text):
    cmd = MagicMock(text=text, user=None, emotion=None, timings={}, stt_confidence=None,
                    stt_vetoed=False, stt_source="primary")
    processor = MagicMock()
    processor.process_command = AsyncMock(return_value=cmd)
    orch = MagicMock()
    orch.process = AsyncMock(return_value=MagicMock(
        intent="domotics", response="ok", success=True, action=None, path=None, timings={},
        was_queued=False, queue_position=None))
    router = RequestRouter(
        command_processor=processor, orchestrator=orch, orchestrator_enabled=True,
        response_handler=response_handler_mock(), audio_manager=MagicMock(), wake_words=("nexa",),
        command_gate=CommandAcceptanceGate(), wake_acoustically_confirmed=True)
    return router, processor, orch


@pytest.mark.asyncio
async def test_false_wake_into_background_speech_costs_no_stt():
    event, closed_after = _false_wake_into_background_speech()
    assert event is not None and event.capture_complete is False
    assert abs(closed_after - CONFIG.max_utterance_s) <= CHUNK_S + 1e-9
    router, processor, orch = _router("prendé la luz del escritorio")
    result = await router.process_command(event)
    assert result["intent"] == "gate_rejected:incomplete_capture"
    processor.process_command.assert_not_awaited()
    orch.process.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_at_the_cap_requires_the_wake_word():
    event, _closed_after = _false_wake_into_background_speech(DISPATCH_AT_MAX)
    assert event is not None and event.capture_complete is True
    router, processor, orch = _router("prendé la luz del escritorio")
    result = await router.process_command(event)
    assert result["intent"] == "gate_rejected:max_utterance_without_wake"
    orch.process.assert_not_awaited()
    event, _closed_after = _false_wake_into_background_speech(DISPATCH_AT_MAX)
    named, _processor, named_orch = _router("Nexa prendé la luz del escritorio")
    result = await named.process_command(event)
    assert not str(result["intent"]).startswith("gate_rejected")
    named_orch.process.assert_awaited()


def test_wake_vad_reaches_the_event_and_reinforces_nothing_below_threshold():
    rng = np.random.default_rng(9)
    before = np.concatenate([_segment(rng, 3.04, 0.003), _segment(rng, 0.48, 0.05)])
    pre_chunks = _chunks(before)
    loop, rs = _loop(_Detector(fire_at=len(pre_chunks) - 1))
    loop._wake_vad_predictors["cocina"] = lambda _audio: 0.42
    process = loop._make_audio_processor(rs)
    for chunk in pre_chunks:
        process(chunk[:, None], CHUNK_SIZE, None, None)
    assert rs.wake_vad == pytest.approx(0.42) and rs.capture_trigger is CaptureTrigger.WAKE
    event = None
    after = np.concatenate([_segment(rng, 1.04, 0.03), _segment(rng, 2.0, 0.003)])
    for index, chunk in enumerate(_chunks(after), start=1):
        process(chunk[:, None], CHUNK_SIZE, None, None)
        event = loop._poll_command(rs)
        if event is not None:
            break
    assert event is not None and event.wake_vad == pytest.approx(0.42)
    assert event.capture_trigger == "wake"
    assert index * CHUNK_S == pytest.approx(1.04 + END_CLOSE_S)


class _SileroHangover:
    """Silero-like predictor whose probability lingers after speech ends."""

    def __init__(self, frames=4):
        self.frames = frames
        self.left = 0

    def __call__(self, audio):
        level = float(np.sqrt(np.mean(np.square(audio))))
        if level > 0.01:
            self.left = self.frames
            return 0.9
        self.left -= 1
        return 0.6 if self.left > 0 else 0.05


def _production_post_wake() -> PostWakeConfig:
    root = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load((root / "config" / "settings.yaml").read_text())
    return PostWakeConfig.from_dict(cfg["rooms"]["endpointing"]["post_wake"])


@pytest.mark.parametrize("vad", [None, _SileroHangover])
def test_production_config_closes_within_one_frame_of_the_legacy_endpoint(vad):
    """No early_command in production (early_dispatch off): no parser-ready close."""
    production = _production_post_wake()
    rng = np.random.default_rng(13)
    before = np.concatenate([_segment(rng, 3.04, 0.002), _segment(rng, 0.48, 0.03)])
    pre_chunks = _chunks(before)
    loop, rs = _loop(_Detector(fire_at=len(pre_chunks) - 1), post_wake=production)
    if vad is not None:
        loop._wake_vad_predictors["cocina"] = vad()
    process = loop._make_audio_processor(rs)
    for chunk in pre_chunks:
        process(chunk[:, None], CHUNK_SIZE, None, None)
    event = None
    for index, chunk in enumerate(_chunks(np.concatenate([_segment(rng, 1.2, 0.03),
                                                          _segment(rng, 3.0, 0.002)])), start=1):
        process(chunk[:, None], CHUNK_SIZE, None, None)
        assert rs.early_command is None
        event = loop._poll_command(rs)
        if event is not None:
            break
    assert event is not None and event.capture_complete is True
    assert loop.endpoint_counts == {"close_silence": 1}
    after_speech = index * CHUNK_S - 1.2
    assert after_speech <= LEGACY_CLOSE_S + CHUNK_S + 1e-9


def test_speech_found_only_by_silero_waits_for_the_longer_silence():
    """Below the level ratio, Silero confirms speech; the close then needs both quiet."""

    def silero(audio):
        return 0.9 if float(np.mean(audio)) > 0.0003 else 0.05

    rng = np.random.default_rng(17)
    before = np.concatenate([_segment(rng, 3.04, 0.0012), _segment(rng, 0.48, 0.0012) + 0.0005])
    pre_chunks = _chunks(before)
    loop, rs = _loop(_Detector(fire_at=len(pre_chunks) - 1))
    loop._wake_vad_predictors["cocina"] = silero
    process = loop._make_audio_processor(rs)
    for chunk in pre_chunks:
        process(chunk[:, None], CHUNK_SIZE, None, None)
    speech = _segment(rng, 1.2, 0.0012) + 0.0005
    assert float(np.sqrt(np.mean(speech ** 2))) < 0.0012 * CONFIG.noise_ratio
    event = None
    for index, chunk in enumerate(_chunks(np.concatenate([speech, _segment(rng, 3.0, 0.0012)])), start=1):
        process(chunk[:, None], CHUNK_SIZE, None, None)
        event = loop._poll_command(rs)
        if event is not None:
            break
    assert event is not None and event.capture_complete is True
    vad_close = math.ceil(CONFIG.vad_end_silence_ms / 1000 / CHUNK_S - 1e-9) * CHUNK_S
    assert index * CHUNK_S == pytest.approx(1.2 + vad_close)


def test_legacy_endpoint_reproduces_the_diagnosis():
    """Without the rule, the Cocina capture closes on the wake word itself."""
    out = _run(floor=0.0012, nexa_rms=0.0035, pause_s=0.24, command_s=1.2, command_rms=0.0032,
               post_wake=None, command_duration=2.0)
    assert out["event"] is not None
    assert out["closed_after"] <= 2 * CHUNK_S
    assert len(out["event"].audio) - out["preroll"] <= 2 * CHUNK_SIZE


def test_parser_ready_keeps_the_fast_close_before_min_listen():
    rng = np.random.default_rng(3)
    detector = _Detector(fire_at=None)
    loop, rs = _loop(detector)
    ready = MagicMock()
    ready.ready_to_dispatch.return_value = True
    rs.listening = True
    rs.command_start_time = time.time()
    rs.capture_trigger = CaptureTrigger.WAKE
    rs.audio_buffer = list(_segment(rng, 0.32, 0.05))
    rs.early_command = ready
    assert loop._poll_command(rs) is None
    rs.audio_buffer.extend(_segment(rng, 0.16, 0.0005))
    event = loop._poll_command(rs)
    assert event is not None and event.capture_complete is True
    assert loop.endpoint_counts == {"close_parser_ready": 1}


@pytest.mark.parametrize("post_wake", [CONFIG, DISPATCH_AT_MAX])
def test_stalled_audio_closes_by_wall_clock(monkeypatch, post_wake):
    detector = _Detector(fire_at=None)
    loop, rs = _loop(detector, post_wake=post_wake)
    rs.listening = True
    rs.command_start_time = 1000.0
    rs.capture_trigger = CaptureTrigger.WAKE
    rs.audio_buffer = list(_segment(np.random.default_rng(1), 0.4, 0.05))
    monkeypatch.setattr(time, "time", lambda: 1000.0 + CONFIG.max_utterance_s + CONFIG.stall_margin_s - 0.01)
    assert loop._poll_command(rs) is None and rs.listening
    monkeypatch.setattr(time, "time", lambda: 1000.0 + CONFIG.max_utterance_s + CONFIG.stall_margin_s + 0.01)
    event = loop._poll_command(rs)
    # A stall closes like the cap: complete only when dispatch at the cap is on.
    assert event is not None and event.capture_complete is post_wake.dispatch_at_max_utterance


# ---------------------------------------------------------------- follow-up start

def _follow_up_run(speech_rms, post_wake=CONFIG):
    rng = np.random.default_rng(11)
    loop, rs = _loop(_Detector(fire_at=None), post_wake=post_wake, follow_up=_follow_up(active=True))
    process = loop._make_audio_processor(rs)
    for chunk in _chunks(_segment(rng, 2.0, 0.0012)):
        process(chunk[:, None], CHUNK_SIZE, None, None)
    assert not rs.listening
    started_at = None
    for index, chunk in enumerate(_chunks(_segment(rng, 1.04, speech_rms)), start=1):
        process(chunk[:, None], CHUNK_SIZE, None, None)
        if rs.listening and started_at is None:
            started_at = index
    return loop, rs, started_at, rng, process


def test_follow_up_starts_on_quiet_speech_relative_to_the_floor():
    loop, rs, started_at, rng, process = _follow_up_run(0.0032)
    assert started_at == 1
    assert rs.capture_trigger is CaptureTrigger.FOLLOW_UP
    # The onset keeps a short pre-roll (follow_up_preroll_s) before the trigger chunk.
    assert rs.wake_sample_index == round(CONFIG.follow_up_preroll_s / CHUNK_S) * CHUNK_SIZE
    event = None
    for chunk in _chunks(_segment(rng, 1.2, 0.0012)):
        process(chunk[:, None], CHUNK_SIZE, None, None)
        event = loop._poll_command(rs)
        if event is not None:
            break
    assert event is not None and event.capture_trigger == "follow_up" and event.capture_complete is True


def test_follow_up_still_requires_human_voice():
    rng = np.random.default_rng(11)
    loop, rs = _loop(_Detector(fire_at=None), follow_up=_follow_up(active=True))
    rs.echo_suppressor.is_human_voice.return_value = False
    process = loop._make_audio_processor(rs)
    for chunk in _chunks(np.concatenate([_segment(rng, 2.0, 0.0012), _segment(rng, 1.0, 0.0032)])):
        process(chunk[:, None], CHUNK_SIZE, None, None)
    assert not rs.listening


def test_follow_up_ignores_ambient_at_the_floor():
    rng = np.random.default_rng(11)
    loop, rs = _loop(_Detector(fire_at=None), follow_up=_follow_up(active=True))
    process = loop._make_audio_processor(rs)
    for chunk in _chunks(_segment(rng, 4.0, 0.0012)):
        process(chunk[:, None], CHUNK_SIZE, None, None)
    assert not rs.listening


def test_follow_up_keeps_its_guard():
    rng = np.random.default_rng(11)
    guard = MagicMock()
    guard.follow_up_allowed.return_value = False
    loop, rs = _loop(_Detector(fire_at=None), follow_up=_follow_up(active=True), ambient_guard=guard)
    process = loop._make_audio_processor(rs)
    for chunk in _chunks(np.concatenate([_segment(rng, 2.0, 0.0012), _segment(rng, 1.0, 0.0032)])):
        process(chunk[:, None], CHUNK_SIZE, None, None)
    assert not rs.listening


def test_legacy_follow_up_start_is_unreachable_for_quiet_speech():
    _loop_, rs, started_at, _rng, _process = _follow_up_run(0.0032, post_wake=None)
    assert started_at is None and not rs.listening


# ---------------------------------------------------------------- units

def test_vad_reinforces_speech_below_the_level_ratio():
    frame = _segment(np.random.default_rng(5), CHUNK_S, 0.0013)
    assert classify_frame(frame, 0.0012, CONFIG, None)[0] is False
    speech, _rms, vad = classify_frame(frame, 0.0012, CONFIG, lambda _a: 0.91)
    assert speech is True and vad == pytest.approx(0.91)
    assert classify_frame(frame, 0.0012, CONFIG, lambda _a: 0.2)[0] is False
    # A failing predictor never makes a frame speech.
    def broken(_a):
        raise RuntimeError("x")
    assert classify_frame(frame, 0.0012, CONFIG, broken)[0] is False
    assert classify_frame(frame, 0.0012, CONFIG, lambda _a: float("nan"))[0] is False


def test_level_ratio_is_relative_to_the_floor_not_absolute():
    frame = _segment(np.random.default_rng(5), CHUNK_S, 0.004)
    assert classify_frame(frame, 0.0012, CONFIG, None)[0] is True
    assert classify_frame(frame, 0.01, CONFIG, None)[0] is False


def test_noise_floor_tracker_percentile_default_and_clamp():
    tracker = NoiseFloorTracker(CONFIG)
    assert tracker.floor() == CONFIG.noise_floor_default_rms
    for value in [0.001] * 20 + [0.05] * 5:
        tracker.observe(value)
    assert tracker.floor() == pytest.approx(0.001)
    quiet = NoiseFloorTracker(CONFIG)
    for _ in range(30):
        quiet.observe(0.0)
    assert quiet.floor() == CONFIG.noise_floor_min_rms
    quiet.observe(float("nan"))
    assert quiet.samples == 30


def test_endpoint_state_requires_confirmed_speech_and_minimum_listen():
    state = EndpointState(wake_index=0, floor=0.001, next_frame=0)
    buffer = list(np.zeros(int(0.08 * SR), dtype=np.float32))
    buffer += list(np.full(int(0.16 * SR), 0.01, dtype=np.float32))
    buffer += list(np.zeros(int(1.2 * SR), dtype=np.float32))
    state.analyze(buffer, CONFIG, None)
    # 160 ms of speech < min_speech_ms: no silence close, no drop (speech seen).
    assert state.decide(CONFIG, SR, False, 0.0) is EndpointDecision.CONTINUE


def test_config_from_settings_enables_the_rule():
    root = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load((root / "config" / "settings.yaml").read_text())
    endpointing = cfg["rooms"]["endpointing"]
    parsed = PostWakeConfig.from_dict(endpointing["post_wake"])
    assert parsed is not None and parsed.max_utterance_s == 5.0 and parsed.min_listen_s == 1.0
    # VC-Q2 pending: the cap keeps the legacy rejection (no STT) by default.
    assert parsed.dispatch_at_max_utterance is False
    with pytest.raises(ValueError):
        PostWakeConfig(dispatch_at_max_utterance="yes")
    assert endpointing["command_duration_s"] == 2.0
    assert PostWakeConfig.from_dict({"enabled": False, "min_listen_s": 1.0}) is None
    assert PostWakeConfig.from_dict(None) is None
    with pytest.raises(ValueError):
        PostWakeConfig.from_dict({"enabled": True, "bogus": 1})
    with pytest.raises(ValueError):
        PostWakeConfig(min_listen_s=6.0)
    with pytest.raises(ValueError):
        PostWakeConfig(noise_ratio=0.5)


def test_main_no_longer_hardcodes_command_duration():
    root = Path(__file__).resolve().parents[3]
    source = (root / "src" / "main.py").read_text()
    assert "command_duration=2.0" not in source
    assert 'PostWakeConfig.from_dict(endpointing_cfg.get("post_wake"))' in source
