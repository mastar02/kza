"""Optional technical lineage of input PCM; never source authority or consent."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import re

from src.evaluation.provenance import (
    CaptureLane, CaptureProvenance, CaptureSource, DSPObservation,
)


_OPENING_REF = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}')
_INPUT_REF = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,61}')


def _dsp_copy(value: DSPObservation) -> DSPObservation:
    return DSPObservation(value.certainty, value.declared_revision, value.observed_revision)


@dataclass(frozen=True)
class CaptureOpening:
    """Observed input format for one opening, without device identifiers.

    source_ref is an opaque opening reference, never a USB path, serial or
    account identity. DSP state may remain unknown or merely declared. The
    actual selection and complete PCM frame count are added by project().
    """

    source: CaptureSource
    source_ref: str
    stream_generation: int
    sample_rate_hz: int
    channels: int
    dsp: DSPObservation = field(default_factory=DSPObservation)

    def __post_init__(self) -> None:
        self._check()
        object.__setattr__(self, 'dsp', _dsp_copy(self.dsp))

    def _check(self) -> None:
        if type(self.source_ref) is not str or _OPENING_REF.fullmatch(self.source_ref) is None:
            raise ValueError('invalid_capture_opening')
        CaptureProvenance(self.source, self.source_ref, CaptureLane.EVALUATION,
            self.stream_generation, 0, self.channels, 0, self.sample_rate_hz, 1, self.dsp)

    def project(self, *, lane: CaptureLane, segment_generation: int, selected_channel: int,
                frame_count: int, input_ref: str) -> CaptureProvenance:
        """Describe the selected owned PCM with its explicit frame-zero anchor.

        input_ref must identify that actual PCM window and cannot contain ':'.
        The final ':' therefore separates opening and input unambiguously.
        Cropped audio and the
        original STT window need different anchors, even if lengths coincide.
        No timestamp, room label or current configuration supplies this fact.
        """
        self._check()
        if type(input_ref) is not str or _INPUT_REF.fullmatch(input_ref) is None:
            raise ValueError('invalid_capture_input_reference')
        return CaptureProvenance(self.source, self.source_ref + ':' + input_ref, lane,
            self.stream_generation, segment_generation, self.channels, selected_channel,
            self.sample_rate_hz, frame_count, _dsp_copy(self.dsp))


def coherent_opening(openings: Iterable[CaptureOpening | None]) -> tuple[CaptureOpening | None, bool]:
    """Return shared metadata and whether known inputs conflict.

    None is an unknown contribution, not an empty accumulator. Callers keep
    their actual chunk sequence so evicting every unknown can restore known
    metadata. Conflicting known openings must be reset before PCM is mixed.
    """
    known = None
    unknown = False
    for opening in openings:
        if opening is None:
            unknown = True
            continue
        if type(opening) is not CaptureOpening:
            raise ValueError('invalid_capture_opening')
        opening._check()
        if known is None:
            known = opening
        elif known != opening:
            return None, True
    return (None if unknown else known), False


_CURRENT_OPENING: ContextVar[CaptureOpening | None] = ContextVar('capture_opening', default=None)


def current_capture_opening() -> CaptureOpening | None:
    """Return only the descriptor bound to this actual packet invocation."""
    return _CURRENT_OPENING.get()


@contextmanager
def capture_opening_context(opening: CaptureOpening | None) -> Iterator[None]:
    """Bind technical metadata for one packet and restore it on every exit."""
    if opening is not None:
        if type(opening) is not CaptureOpening:
            raise ValueError('invalid_capture_opening')
        opening._check()
    token = _CURRENT_OPENING.set(opening)
    try:
        yield
    finally:
        _CURRENT_OPENING.reset(token)
