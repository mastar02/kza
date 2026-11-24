"""Unit tests for src/audio/clip_io.py (shared channel selection + private audio I/O)."""
from __future__ import annotations

import stat

import numpy as np
import pytest
import soundfile as sf

from src.audio.clip_io import (
    ArchiveChannelMode,
    resample_linear_16k,
    select_archive_audio,
    select_mono_channel,
    write_private_audio,
    write_private_bytes,
)


def test_select_archive_audio_modes_and_fallbacks():
    ch0 = np.full(160, 0.1, dtype=np.float32)
    ch1 = np.full(160, 0.5, dtype=np.float32)
    stereo = np.stack([ch0, ch1], axis=1)

    assert np.allclose(select_archive_audio(stereo, ArchiveChannelMode.MONO_CH0), ch0)
    assert np.allclose(select_archive_audio(stereo, ArchiveChannelMode.MONO_CH1), ch1)
    both = select_archive_audio(stereo, ArchiveChannelMode.STEREO_2CH)
    assert both.shape == (160, 2)
    assert np.allclose(both[:, 0], ch0) and np.allclose(both[:, 1], ch1)

    # 1D input degrades cleanly across all modes
    assert select_archive_audio(ch0, ArchiveChannelMode.STEREO_2CH).ndim == 1
    assert select_archive_audio(ch0, ArchiveChannelMode.MONO_CH1).ndim == 1

    with pytest.raises(ValueError):
        select_archive_audio(np.zeros((0,), dtype=np.float32))


def test_select_mono_channel_returns_selected_index():
    stereo = np.stack([np.zeros(80, dtype=np.float32), np.ones(80, dtype=np.float32)], axis=1)
    mono0, idx0 = select_mono_channel(stereo, 0)
    mono1, idx1 = select_mono_channel(stereo, 1)
    mono_oob, idx_oob = select_mono_channel(stereo, 5)
    assert idx0 == 0 and np.all(mono0 == 0.0)
    assert idx1 == 1 and np.all(mono1 == 1.0)
    assert idx_oob == 0 and np.all(mono_oob == 0.0)


def test_resample_and_private_writes(tmp_path):
    src = np.ones((4800, 2), dtype=np.float32) * 0.25
    res = resample_linear_16k(src, 48000)
    assert res.shape == (1600, 2)

    raw_file = tmp_path / "priv" / "meta.json"
    write_private_bytes(raw_file, b"{}")
    assert stat.S_IMODE(raw_file.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(raw_file.stat().st_mode) == 0o600

    flac_file = tmp_path / "priv" / "clip.flac"
    write_private_audio(flac_file, res, 16000, fmt="FLAC")
    assert stat.S_IMODE(flac_file.stat().st_mode) == 0o600
    data, sr = sf.read(str(flac_file), dtype="float32")
    assert sr == 16000 and data.shape == (1600, 2)
