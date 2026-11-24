"""Shared private clip I/O and explicit channel-selection primitives.

Centralizes safe ``0700``/``0600`` audio writing (FLAC/WAV) and explicit
channel layout selection (``mono_ch0``, ``mono_ch1``, ``stereo_2ch``) so that
``AudioArchiver``, ``tools/stt_ab_compare.py``, and dataset exporters follow a
single contract instead of duplicating channel slicing and permissions logic.
"""
from __future__ import annotations

from enum import StrEnum
import os
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16000


class ArchiveChannelMode(StrEnum):
    """Explicit channel retention mode for recorded/archived audio clips."""

    MONO_CH0 = "mono_ch0"
    MONO_CH1 = "mono_ch1"
    STEREO_2CH = "stereo_2ch"


def select_archive_audio(
    audio: np.ndarray,
    channel_mode: ArchiveChannelMode | str = ArchiveChannelMode.MONO_CH0,
) -> np.ndarray:
    """Extract the configured channel layout from a 1D or 2D audio segment.

    Args:
        audio: Raw segment samples, shape ``(frames,)`` or ``(frames, channels)``.
        channel_mode: ``mono_ch0`` (Conference), ``mono_ch1`` (ASR), or
            ``stereo_2ch`` (preserve ``ch0`` + ``ch1``). Falls back cleanly to
            channel 0 when the input has fewer channels than requested.

    Returns:
        Contiguous array ready for encoding.

    Raises:
        ValueError: If ``audio`` is empty, has unsupported dimensions, or
            ``channel_mode`` is unknown.
    """
    mode = ArchiveChannelMode(channel_mode)
    if audio.ndim not in (1, 2) or audio.size == 0 or audio.shape[0] == 0:
        raise ValueError("audio vacío")
    if audio.ndim == 1:
        return np.ascontiguousarray(audio)
    channels = audio.shape[1]
    if channels < 1:
        raise ValueError("audio vacío")
    if mode is ArchiveChannelMode.STEREO_2CH:
        if channels >= 2:
            return np.ascontiguousarray(audio[:, :2])
        return np.ascontiguousarray(audio[:, 0])
    if mode is ArchiveChannelMode.MONO_CH1:
        col = 1 if channels >= 2 else 0
        return np.ascontiguousarray(audio[:, col])
    return np.ascontiguousarray(audio[:, 0])


def select_mono_channel(audio: np.ndarray, channel: int = 0) -> tuple[np.ndarray, int]:
    """Select one mono channel with safe fallback to channel 0 for 1D/narrow audio.

    Args:
        audio: 1D ``(frames,)`` or 2D ``(frames, channels)`` array.
        channel: Preferred channel index (``0`` = Conference, ``1`` = ASR).

    Returns:
        ``(mono_array, selected_channel_index)``.
    """
    if audio.ndim == 1:
        return np.ascontiguousarray(audio), 0
    if audio.ndim != 2 or audio.shape[1] < 1:
        raise ValueError("invalid_audio_shape")
    selected = channel if 0 <= channel < audio.shape[1] else 0
    return np.ascontiguousarray(audio[:, selected]), selected


def resample_linear_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample 1D or 2D float32 PCM linearly to 16 kHz if needed."""
    if sample_rate == TARGET_SAMPLE_RATE or samples.shape[0] == 0:
        return np.ascontiguousarray(samples, dtype=np.float32)
    target_len = max(1, int(round(samples.shape[0] * TARGET_SAMPLE_RATE / sample_rate)))
    idx = np.linspace(0, samples.shape[0] - 1, target_len)
    base = np.arange(samples.shape[0])
    if samples.ndim == 1:
        return np.interp(idx, base, samples).astype(np.float32)
    cols = [np.interp(idx, base, samples[:, c]).astype(np.float32) for c in range(samples.shape[1])]
    return np.ascontiguousarray(np.stack(cols, axis=1))


def write_private_bytes(path: Path, data: bytes) -> None:
    """Write bytes to a private ``0600`` file inside a ``0700`` directory."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
    finally:
        os.close(fd)


def write_private_audio(
    path: Path,
    samples: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    fmt: str = "FLAC",
    subtype: str | None = None,
) -> None:
    """Write mono or multichannel PCM to a ``0600`` file (FLAC or WAV)."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=False) as handle:
            kwargs = {"format": fmt}
            if subtype is not None:
                kwargs["subtype"] = subtype
            sf.write(handle, samples, sample_rate, **kwargs)
    finally:
        os.close(fd)
