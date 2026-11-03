"""Declared provenance of one owned PCM segment; no runtime discovery.

This DTO never serializes CaptureObservation, device paths or USB properties.
Adapters supply an opaque source reference and observations made for the actual
PCM segment. Configuration supports DECLARED, never OBSERVED DSP state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re


MAX_COUNTER = (1 << 63) - 1


class CaptureSource(str, Enum):
    """Origin category, independent from recording consent or retention."""

    LIVE_INPUT = "live_input"
    CONSENTED_RECORDING = "consented_recording"


class CaptureLane(str, Enum):
    """Processing branch that selected the described PCM."""

    COMMAND = "command"
    AMBIENT = "ambient"
    EVALUATION = "evaluation"


class DSPCertainty(str, Enum):
    """Whether DSP state is unknown, configured or independently observed."""

    UNKNOWN = "unknown"
    DECLARED = "declared"
    OBSERVED = "observed"


def _opaque(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is None:
        raise ValueError("invalid_provenance_reference")


def _counter(value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= MAX_COUNTER:
        raise ValueError("invalid_provenance_counter")


@dataclass(frozen=True)
class DSPObservation:
    """Separate intended configuration from evidence of applied DSP state."""

    certainty: DSPCertainty = DSPCertainty.UNKNOWN
    declared_revision: str | None = None
    observed_revision: str | None = None

    def __post_init__(self) -> None:
        if type(self.certainty) is not DSPCertainty:
            raise ValueError("invalid_dsp_certainty")
        if self.declared_revision is not None:
            _opaque(self.declared_revision)
        if self.observed_revision is not None:
            _opaque(self.observed_revision)
        if self.certainty is DSPCertainty.UNKNOWN:
            if self.declared_revision is not None or self.observed_revision is not None:
                raise ValueError("unknown_dsp_has_no_revision")
        elif self.certainty is DSPCertainty.DECLARED:
            if self.declared_revision is None or self.observed_revision is not None:
                raise ValueError("declared_dsp_is_not_observed")
        elif self.observed_revision is None:
            raise ValueError("observed_dsp_requires_revision")


@dataclass(frozen=True)
class CaptureProvenance:
    """One selected PCM source; no inferred identity or wall-clock interval.

    stream_generation identifies source opening; segment_generation identifies
    the captured segment in that stream. sample_rate_hz/frame_count describe the
    owned selected PCM whose frame zero anchors every FrameInterval.
    """

    source: CaptureSource
    source_ref: str
    lane: CaptureLane
    stream_generation: int
    segment_generation: int
    source_channels: int
    selected_channel: int
    sample_rate_hz: int
    frame_count: int
    dsp: DSPObservation = field(default_factory=DSPObservation)

    def __post_init__(self) -> None:
        if type(self.source) is not CaptureSource or type(self.lane) is not CaptureLane:
            raise ValueError("invalid_capture_source_or_lane")
        _opaque(self.source_ref)
        _counter(self.stream_generation)
        _counter(self.segment_generation)
        if type(self.source_channels) is not int or not 1 <= self.source_channels <= 64:
            raise ValueError("invalid_source_channels")
        if type(self.selected_channel) is not int or not 0 <= self.selected_channel < self.source_channels:
            raise ValueError("invalid_selected_channel")
        if type(self.sample_rate_hz) is not int or not 8_000 <= self.sample_rate_hz <= 192_000:
            raise ValueError("invalid_pcm_sample_rate")
        _counter(self.frame_count, minimum=1)
        if type(self.dsp) is not DSPObservation:
            raise ValueError("invalid_dsp_observation")
        DSPObservation.__post_init__(self.dsp)

    @property
    def comparable(self) -> bool:
        """A stable DSP cohort needs observation, not configuration intent."""
        return self.dsp.certainty is DSPCertainty.OBSERVED


@dataclass(frozen=True)
class FrameInterval:
    """Half-open interval anchored at frame zero of the owned capture PCM."""

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        _counter(self.start_frame)
        _counter(self.end_frame, minimum=1)
        if self.start_frame >= self.end_frame:
            raise ValueError("invalid_frame_interval")

    def millisecond_bounds(self, sample_rate_hz: int) -> tuple[int, int]:
        """Compatibility bounds; comparison uses exact frames, not rounding."""
        if type(sample_rate_hz) is not int or not 8_000 <= sample_rate_hz <= 192_000:
            raise ValueError("invalid_pcm_sample_rate")
        return (self.start_frame * 1000 // sample_rate_hz,
                (self.end_frame * 1000 + sample_rate_hz - 1) // sample_rate_hz)
