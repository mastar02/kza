"""Post-wake listening rule for command captures (V1, voice-diag 2026-11-17).

The legacy endpoint measured its 2 s timeout from the start of the 0.96 s
pre-roll and closed on an absolute RMS of 0.015, so most captures closed
~50 ms after the wake on the wake word itself, and quiet rooms (Cocina speech
RMS 0.003-0.006) never reached the follow-up start level of 0.02.

This module measures every duration in captured samples after the wake and
decides on speech relative to the room's own measured noise floor, optionally
reinforced by the per-room Silero VAD that the loop already owns. It holds no
audio beyond the frame being classified and never sees transcripts.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, fields
from enum import Enum, StrEnum
from typing import Callable

import numpy as np


class CaptureTrigger(StrEnum):
    """What opened a command capture."""

    WAKE = "wake"
    FOLLOW_UP = "follow_up"
    BARGE_IN = "barge_in"
    INLINE = "inline"
    UNKNOWN = "unknown"


class EndpointDecision(Enum):
    """Outcome of one endpoint evaluation."""

    CONTINUE = "continue"
    CLOSE_SILENCE = "close_silence"
    CLOSE_READY = "close_parser_ready"
    CLOSE_MAX = "close_max_utterance"
    DROP_NO_SPEECH = "drop_no_speech"

    @property
    def closes(self) -> bool:
        """True when the capture ends (dispatch or drop)."""
        return self is not EndpointDecision.CONTINUE

    @property
    def dispatches(self) -> bool:
        """True when the captured audio goes to STT."""
        return self in (EndpointDecision.CLOSE_SILENCE, EndpointDecision.CLOSE_READY,
                        EndpointDecision.CLOSE_MAX)


@dataclass(frozen=True)
class PostWakeConfig:
    """Tunable post-wake listening rule (``rooms.endpointing.post_wake``).

    Attributes:
        min_listen_s: Minimum audio after the wake before a silence close.
        max_utterance_s: Hard cap of audio after the wake.
        no_speech_timeout_s: Without confirmed speech (``min_speech_ms``)
            after the wake for this long, the capture is dropped without STT;
            a wake-word tail or a click alone does not keep it open.
        end_silence_ms: Sustained level-only silence (RMS below
            ``noise_ratio`` x floor, Silero ignored) after speech confirmed by
            level that ends the command. Silero is ignored here because its
            probability lingers after speech ends and stretched every close.
        vad_end_silence_ms: Silence (neither level nor Silero) that ends a
            command whose speech was confirmed only with Silero's help.
        ready_silence_ms: Shorter silence accepted when the streaming parser
            already holds a complete command (keeps the fast close).
        min_speech_ms: Speech after the wake needed to confirm a command
            (the wake word tail alone must not count).
        noise_ratio: A frame is speech when its RMS reaches this multiple of
            the room's noise floor.
        noise_floor_min_rms: Lower clamp of the measured floor (digital
            silence must not make every noise "speech").
        noise_floor_default_rms: Floor used before enough ambient history.
        noise_window_s: Ambient history used to measure the floor.
        noise_percentile: Percentile of the ambient chunk RMS taken as floor.
        noise_min_history_s: Ambient history required before trusting it.
        vad_threshold: Silero probability that also marks a frame as speech.
            ``None`` disables the VAD reinforcement (level only).
        frame_samples: Analysis frame (80 ms at 16 kHz, one capture chunk).
        stall_margin_s: Wall-clock margin past ``max_utterance_s`` that closes
            a capture whose audio stopped arriving.
        follow_up_preroll_s: Audio kept before a follow-up speech onset.
        dispatch_at_max_utterance: Send a capture that reaches
            ``max_utterance_s`` with speech to STT as complete. False (default,
            pending the user's answer to VC-Q2) keeps the legacy rejection:
            the capture is marked incomplete and the router rejects it as
            ``gate_rejected:incomplete_capture`` without STT. True dispatches
            it, and the router then requires a wake word in the transcript.
    """

    min_listen_s: float = 1.0
    max_utterance_s: float = 5.0
    no_speech_timeout_s: float = 2.5
    end_silence_ms: int = 300
    vad_end_silence_ms: int = 600
    ready_silence_ms: int = 150
    min_speech_ms: int = 240
    noise_ratio: float = 1.8
    noise_floor_min_rms: float = 0.0006
    noise_floor_default_rms: float = 0.002
    noise_window_s: float = 10.0
    noise_percentile: float = 20.0
    noise_min_history_s: float = 1.0
    vad_threshold: float | None = 0.5
    frame_samples: int = 1280
    stall_margin_s: float = 1.0
    follow_up_preroll_s: float = 0.32
    dispatch_at_max_utterance: bool = False

    def __post_init__(self) -> None:
        if type(self.dispatch_at_max_utterance) is not bool:
            raise ValueError("invalid post_wake.dispatch_at_max_utterance")
        positive = ("min_listen_s", "max_utterance_s", "no_speech_timeout_s", "end_silence_ms",
                    "vad_end_silence_ms",
                    "ready_silence_ms", "min_speech_ms", "noise_ratio", "noise_floor_min_rms",
                    "noise_floor_default_rms", "noise_window_s", "frame_samples", "stall_margin_s")
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid post_wake.{name}")
        for name in ("noise_percentile", "noise_min_history_s", "follow_up_preroll_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid post_wake.{name}")
        if self.noise_percentile > 100:
            raise ValueError("invalid post_wake.noise_percentile")
        if self.noise_ratio < 1.0:
            raise ValueError("invalid post_wake.noise_ratio")
        if not (self.min_listen_s <= self.max_utterance_s and self.no_speech_timeout_s <= self.max_utterance_s):
            raise ValueError("invalid post_wake window ordering")
        if self.vad_threshold is not None and not (0.0 < float(self.vad_threshold) <= 1.0):
            raise ValueError("invalid post_wake.vad_threshold")
        if int(self.frame_samples) != self.frame_samples:
            raise ValueError("invalid post_wake.frame_samples")

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "PostWakeConfig | None":
        """Build from ``rooms.endpointing.post_wake``; ``enabled: false`` → None.

        Args:
            cfg: The config mapping, or None.

        Returns:
            The validated config, or None to keep the legacy endpoint.
        """
        cfg = dict(cfg or {})
        if not cfg.pop("enabled", False):
            return None
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(f"unknown post_wake keys: {unknown}")
        return cls(**cfg)


class NoiseFloorTracker:
    """Rolling ambient RMS of one room, measured while it is not capturing."""

    def __init__(self, config: PostWakeConfig, sample_rate: int = 16000, chunk_samples: int = 1280):
        self._config = config
        chunk_s = chunk_samples / sample_rate
        self._history: deque[float] = deque(maxlen=max(1, int(round(config.noise_window_s / chunk_s))))
        self._min_history = max(1, int(round(config.noise_min_history_s / chunk_s)))
        self._cached: float | None = None
        self._since_refresh = 0

    def observe(self, rms: float) -> None:
        """Record the RMS of one ambient chunk."""
        if not math.isfinite(rms) or rms < 0:
            return
        self._history.append(float(rms))
        self._since_refresh += 1
        if self._since_refresh >= 8:
            self._cached = None

    def floor(self) -> float:
        """Current floor estimate (clamped, default before enough history)."""
        if len(self._history) < self._min_history:
            return max(self._config.noise_floor_default_rms, self._config.noise_floor_min_rms)
        if self._cached is None:
            value = float(np.percentile(np.fromiter(self._history, dtype=np.float64),
                                        self._config.noise_percentile))
            self._cached = max(value, self._config.noise_floor_min_rms)
            self._since_refresh = 0
        return self._cached

    @property
    def samples(self) -> int:
        """Number of ambient chunks currently retained."""
        return len(self._history)


def frame_rms(frame) -> float:
    """RMS of a mono float frame."""
    array = np.asarray(frame, dtype=np.float32)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))


def classify_frame(frame, floor: float, config: PostWakeConfig,
                   vad_predict: Callable[[np.ndarray], float] | None) -> tuple[bool, float, float | None]:
    """Decide whether one frame is speech.

    Speech when the level reaches ``noise_ratio`` times the floor, or when the
    Silero probability reaches ``vad_threshold``. Silence requires both to say
    so: the XVF3800 output often reads low on Silero for far-field speech, and
    the level test alone misses speech in a room whose floor is speech.

    Args:
        frame: Mono float samples.
        floor: Room noise floor RMS for this capture.
        config: Post-wake config.
        vad_predict: Per-room Silero predictor, or None.

    Returns:
        (is_speech, rms, vad probability or None when not evaluated).
    """
    array = np.ascontiguousarray(np.asarray(frame, dtype=np.float32))
    rms = frame_rms(array)
    if rms >= floor * config.noise_ratio:
        return True, rms, None
    if vad_predict is None or config.vad_threshold is None:
        return False, rms, None
    try:
        vad = float(vad_predict(array))
    except Exception:
        return False, rms, None
    if not math.isfinite(vad):
        return False, rms, None
    return vad >= config.vad_threshold, rms, vad


@dataclass
class EndpointState:
    """Per-capture endpoint bookkeeping (sample indexes into the capture buffer).

    Attributes:
        wake_index: Buffer index of the first sample after the wake.
        floor: Noise floor frozen at capture start.
        next_frame: Buffer index of the next unclassified frame.
        speech_samples: Speech samples classified after the wake.
        speech_frames: Speech frames classified after the wake.
        last_speech_end: Buffer index just past the last speech frame.
        level_speech_samples: Speech samples that passed the level test.
        last_level_end: Buffer index just past the last level speech frame.
        vad_max: Highest Silero probability evaluated in this capture.
        peak_rms: Highest frame RMS after the wake.
        trigger: What opened the capture.
        decision: The closing decision, once the capture closed.
    """

    wake_index: int
    floor: float
    next_frame: int
    trigger: CaptureTrigger = CaptureTrigger.UNKNOWN
    speech_samples: int = 0
    speech_frames: int = 0
    last_speech_end: int | None = None
    level_speech_samples: int = 0
    last_level_end: int | None = None
    vad_max: float | None = None
    peak_rms: float = 0.0
    decision: EndpointDecision | None = None

    def analyze(self, buffer, config: PostWakeConfig,
                vad_predict: Callable[[np.ndarray], float] | None) -> None:
        """Classify every complete, not yet classified frame of ``buffer``."""
        size = config.frame_samples
        while self.next_frame + size <= len(buffer):
            start = self.next_frame
            speech, rms, vad = classify_frame(buffer[start:start + size], self.floor, config, vad_predict)
            self.peak_rms = max(self.peak_rms, rms)
            if vad is not None:
                self.vad_max = vad if self.vad_max is None else max(self.vad_max, vad)
            if speech:
                self.speech_frames += 1
                self.speech_samples += size
                self.last_speech_end = start + size
                if rms >= self.floor * config.noise_ratio:
                    self.level_speech_samples += size
                    self.last_level_end = start + size
            self.next_frame = start + size

    def decide(self, config: PostWakeConfig, sample_rate: int, parser_ready: bool,
               wall_since_wake_s: float) -> EndpointDecision:
        """Apply the post-wake rule to the classified audio.

        Args:
            config: Post-wake config.
            sample_rate: Capture sample rate.
            parser_ready: The streaming parser holds a complete command.
            wall_since_wake_s: Wall-clock time since the wake (stall guard only).

        Returns:
            The endpoint decision.
        """
        post_s = (self.next_frame - self.wake_index) / sample_rate
        speech_s = self.speech_samples / sample_rate
        confirmed = speech_s * 1000 >= config.min_speech_ms
        silence_s = ((self.next_frame - self.last_speech_end) / sample_rate
                     if self.last_speech_end is not None else None)
        if parser_ready and self.speech_frames and silence_s is not None \
                and silence_s * 1000 >= config.ready_silence_ms:
            return EndpointDecision.CLOSE_READY
        if confirmed and post_s >= config.min_listen_s:
            if self.level_speech_samples * 1000 >= config.min_speech_ms * sample_rate:
                # Level-confirmed speech: level-only silence, Silero hangover ignored.
                quiet_ms = (self.next_frame - self.last_level_end) * 1000 / sample_rate
                needed_ms = config.end_silence_ms
            else:
                quiet_ms = silence_s * 1000
                needed_ms = config.vad_end_silence_ms
            if quiet_ms >= needed_ms:
                return EndpointDecision.CLOSE_SILENCE
        if post_s >= config.max_utterance_s or wall_since_wake_s >= config.max_utterance_s + config.stall_margin_s:
            return EndpointDecision.CLOSE_MAX if confirmed else EndpointDecision.DROP_NO_SPEECH
        if not confirmed and post_s >= config.no_speech_timeout_s:
            return EndpointDecision.DROP_NO_SPEECH
        return EndpointDecision.CONTINUE

    def summary(self, sample_rate: int) -> dict:
        """Numbers-only description for logs (no audio, no text)."""
        return {
            "trigger": self.trigger.value,
            "post_wake_s": round((self.next_frame - self.wake_index) / sample_rate, 2),
            "speech_s": round(self.speech_samples / sample_rate, 2),
            "floor": round(self.floor, 4),
            "peak_rms": round(self.peak_rms, 4),
            "vad_max": None if self.vad_max is None else round(self.vad_max, 2),
        }
