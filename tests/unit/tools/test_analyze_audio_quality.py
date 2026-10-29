"""Synthetic faults distinguish saturation, silence, duplication and channel mixups."""

import numpy as np
import soundfile as sf

from tools.analyze_audio_quality import measure_channel, measure_file, summarize


def test_clean_sine_does_not_report_clipping_dropouts_or_repeated_blocks():
    rate = 16000
    signal = .1 * np.sin(2 * np.pi * 997 * np.arange(rate) / rate)
    measured = measure_channel(signal, rate)
    assert abs(measured['rms_dbfs'] + 23.0103) < .001
    assert measured['near_full_scale_fraction'] == 0
    assert measured['longest_internal_exact_zero_run_ms'] < 5
    assert measured['repeated_adjacent_nonzero_80ms_blocks'] == 0
    assert abs(measured['spectrum']['strongest_bin_hz'] - 997) < 16


def test_clipped_signal_and_internal_dropout_are_measured_separately():
    rate = 16000
    signal = np.clip(2 * np.sin(2 * np.pi * 997 * np.arange(rate) / rate), -1, 1)
    signal[8000:8160] = 0
    measured = measure_channel(signal, rate)
    assert measured['near_full_scale_fraction'] > .5
    assert measured['longest_near_full_scale_run_samples'] >= 3
    assert measured['longest_internal_exact_zero_run_ms'] == 10


def test_repeated_nonzero_block_is_detected_but_silence_is_not_replay():
    block = np.random.default_rng(1).normal(0, .01, 1280)
    assert measure_channel(np.tile(block, 4), 16000)['repeated_adjacent_nonzero_80ms_blocks'] == 3
    silent = measure_channel(np.zeros(16000), 16000)
    assert silent['repeated_adjacent_nonzero_80ms_blocks'] == 0
    assert silent['rms_dbfs'] is None and silent['spectrum'] is None


def test_file_preserves_channels_and_summary_does_not_hide_silent_channel(tmp_path):
    signal = np.column_stack((np.full(16000, .25), np.zeros(16000)))
    path = tmp_path / '20261029-130000-123_synthetic-room_0.50.wav'
    sf.write(path, signal, 16000, subtype='PCM_16')
    measured = measure_file(path)
    assert measured['channels'] == 2 and measured['room'] == 'synthetic-room'
    assert measured['measurements'][0]['rms_dbfs'] is not None
    assert measured['measurements'][1]['rms_dbfs'] is None
    summary = summarize([measured])
    assert summary['decoded_files'] == 1
    assert summary['groups'][0]['all_zero_channels'] == 1
    assert summary['groups'][0]['duration_s'] == 1
