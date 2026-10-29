"""Measure local recordings without changing source files or inferring speech quality.

Input is a JSON inventory containing objects with a ``path`` field. Output
contains private paths and must stay outside Git. Metrics describe waveforms;
they do not establish acoustic SNR, intelligibility or a hardware root cause.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import soundfile as sf


def _db(value: float) -> float | None:
    return 20 * math.log10(value) if value > 0 else None


def _longest_run(mask: np.ndarray) -> int:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(np.max(ends - starts)) if starts.size else 0


def measure_channel(audio: np.ndarray, sample_rate: int) -> dict:
    """Measure one normalized mono waveform, keeping silent cases explicit."""
    if audio.ndim != 1 or not audio.size or sample_rate <= 0 or not np.isfinite(audio).all():
        raise ValueError('invalid_audio_signal')
    audio = audio.astype(np.float64, copy=False)
    absolute = np.abs(audio)
    peak = float(absolute.max())
    rms = float(np.sqrt(np.mean(audio * audio)))
    nonzero = np.flatnonzero(audio)
    active = audio[nonzero[0]:nonzero[-1] + 1] if nonzero.size else audio[:0]
    differences = np.abs(np.diff(audio))
    near_full = absolute >= 0.9999
    frame_size = max(1, round(sample_rate * .02))
    count = audio.size // frame_size
    frame_rms = np.sqrt(np.mean(audio[:count * frame_size].reshape(count, frame_size) ** 2, axis=1)) if count else np.array([rms])
    frame_percentiles = np.percentile(frame_rms, [10, 50, 90])
    positive_frames = frame_rms[frame_rms > 0]
    percentile_span = None
    if positive_frames.size and frame_percentiles[0] > 0:
        percentile_span = _db(float(frame_percentiles[2] / frame_percentiles[0]))
    spectrum = None
    # Welch-style averaged periodogram. The window length follows sample rate
    # so frequency bins are about 15.6 Hz; do not compare raw FFT amplitudes.
    window_size = 2 ** max(5, int(round(math.log2(sample_rate / 15.625))))
    if audio.size >= window_size and peak > 0:
        windows = np.lib.stride_tricks.sliding_window_view(audio, window_size)[::window_size // 2]
        centered = windows - windows.mean(axis=1, keepdims=True)
        power = np.mean(np.abs(np.fft.rfft(centered * np.hanning(window_size), axis=1)) ** 2, axis=0)
        frequencies = np.fft.rfftfreq(window_size, 1 / sample_rate)
        total = float(power.sum())
        bands = {}
        for name, lower, upper in [('below_100_hz', 0, 100), ('100_1000_hz', 100, 1000),
                                   ('1000_4000_hz', 1000, 4000), ('4000_6000_hz', 4000, 6000),
                                   ('above_6000_hz', 6000, sample_rate / 2 + 1)]:
            bands[name] = float(power[(frequencies >= lower) & (frequencies < upper)].sum() / total) if total else 0.
        strongest = int(np.argmax(power[1:]) + 1)
        spectrum = {'energy_fractions': bands, 'strongest_bin_hz': float(frequencies[strongest])}
    block_size = round(sample_rate * .08)
    repeated_nonzero = 0
    if audio.size >= block_size * 2:
        blocks = audio[:audio.size // block_size * block_size].reshape(-1, block_size)
        repeated_nonzero = int(np.sum(np.all(blocks[1:] == blocks[:-1], axis=1) & np.any(blocks[1:] != 0, axis=1)))
    return {
        'samples': int(audio.size), 'duration_s': audio.size / sample_rate,
        'peak_dbfs': _db(peak), 'rms_dbfs': _db(rms), 'dc_mean': float(audio.mean()),
        'crest_db': _db(peak / rms) if rms else None,
        'near_full_scale_fraction': float(np.mean(near_full)),
        'longest_near_full_scale_run_samples': _longest_run(near_full),
        'exact_zero_fraction': float(np.mean(audio == 0)),
        'longest_exact_zero_run_ms': _longest_run(audio == 0) / sample_rate * 1000,
        'longest_internal_exact_zero_run_ms': _longest_run(active == 0) / sample_rate * 1000 if active.size else None,
        'max_adjacent_jump': float(differences.max()) if differences.size else 0.,
        'adjacent_jumps_over_025': int(np.sum(differences > .25)),
        'frame_rms_dbfs_p10_p50_p90': [_db(float(value)) for value in frame_percentiles],
        'frame_level_span_p90_p10_db': percentile_span,
        'repeated_adjacent_nonzero_80ms_blocks': repeated_nonzero,
        'spectrum': spectrum,
    }


def measure_file(path: Path) -> dict:
    """Decode a file read-only and measure each channel without downmixing."""
    with sf.SoundFile(path, mode='r') as recording:
        sample_rate, subtype = recording.samplerate, recording.subtype
        audio = recording.read(dtype='float64', always_2d=True)
    match = re.fullmatch(r'(\d{8})-(\d{6})-\d{3}_(.+)_-?\d+(?:\.\d+)?\.wav', path.name)
    room, date = (match[3], match[1]) if match else (None, None)
    if 'captured' in path.parts:
        bucket = 'rejected' if 'rejected' in path.parts else 'missed' if 'missed' in path.parts else 'accepted'
        kind = 'historical_wake_capture'
    elif 'wakeword_training' in path.parts:
        kind, bucket = 'manual_training', None
    elif 'earcons' in path.parts:
        kind, bucket = 'notification', None
    else:
        kind, bucket = 'unclassified_audio', None
    return {'path': str(path), 'kind': kind, 'bucket': bucket, 'room': room, 'filename_date': date,
            'sample_rate': sample_rate, 'subtype': subtype, 'channels': audio.shape[1],
            'pcm_sha256': hashlib.sha256(audio.tobytes()).hexdigest(),
            'measurements': [measure_channel(audio[:, index], sample_rate) for index in range(audio.shape[1])]}


def summarize(records: list[dict]) -> dict:
    """Aggregate measurements without publishing filenames or recording text."""
    groups = defaultdict(list)
    for row in records:
        if 'error_type' not in row:
            groups[(row['kind'], row['room'])].append(row)
    result = []
    for (kind, room), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        measurements = [channel for row in rows for channel in row['measurements']]
        def percentiles(field):
            values = [measurement[field] for measurement in measurements if measurement[field] is not None]
            return [float(value) for value in np.percentile(values, [10, 50, 90])] if values else None
        result.append({
            'kind': kind, 'room': room, 'files': len(rows),
            'duration_s': sum(row['measurements'][0]['duration_s'] for row in rows),
            'sample_rates': dict(Counter(row['sample_rate'] for row in rows)),
            'channels': dict(Counter(row['channels'] for row in rows)),
            'subtypes': dict(Counter(row['subtype'] for row in rows)),
            'filename_dates': sorted({row['filename_date'] for row in rows if row['filename_date']}),
            'channel_count': len(measurements),
            'rms_dbfs_p10_p50_p90': percentiles('rms_dbfs'),
            'peak_dbfs_p10_p50_p90': percentiles('peak_dbfs'),
            'crest_db_p10_p50_p90': percentiles('crest_db'),
            'frame_level_span_db_p10_p50_p90': percentiles('frame_level_span_p90_p10_db'),
            'channels_with_near_full_scale_samples': sum(m['near_full_scale_fraction'] > 0 for m in measurements),
            'channels_with_3_consecutive_near_full_scale_samples': sum(m['longest_near_full_scale_run_samples'] >= 3 for m in measurements),
            'all_zero_channels': sum(m['exact_zero_fraction'] == 1 for m in measurements),
            'channels_with_internal_exact_zero_run_at_least_5ms': sum((m['longest_internal_exact_zero_run_ms'] or 0) >= 5 for m in measurements),
            'channels_with_repeated_nonzero_80ms_block': sum(m['repeated_adjacent_nonzero_80ms_blocks'] > 0 for m in measurements),
            'channels_with_adjacent_jump_over_025': sum(m['adjacent_jumps_over_025'] > 0 for m in measurements),
            'above_6000_hz_energy_fraction_p10_p50_p90': [float(value) for value in np.percentile([
                m['spectrum']['energy_fractions']['above_6000_hz'] for m in measurements if m['spectrum']], [10, 50, 90])]
                if any(m['spectrum'] for m in measurements) else None,
        })
    limits = ['No acoustic SNR or perceptual quality score is inferred.',
              'Exact zero runs, jumps, spectral bands and near-full-scale samples are indicators requiring context.']
    if any(kind == 'historical_wake_capture' for kind, _ in groups):
        limits.extend(['Historical room and capture date come from filenames, not verified runtime identity.',
                       'Historical wake captures do not establish the content or channel of current dashboard previews.'])
    if any(kind == 'current_dashboard_preview' for kind, _ in groups):
        limits.extend(['Current room labels come from the authenticated API; physical microphone binding is not independently verified.',
                       'Preview metadata does not distinguish ambient conversation-channel audio from command ASR-channel audio.'])
    return {'attempted_files': len(records), 'decoded_files': sum('error_type' not in row for row in records),
            'decode_errors': dict(Counter(row['error_type'] for row in records if 'error_type' in row)),
            'groups': result, 'limits': limits}


def main() -> None:
    """Write a private per-file report and an aggregate summary from an inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    records, seen = [], set()
    inventory = json.loads(args.inventory.read_text())
    with (args.output / 'measurements.jsonl').open('w') as output:
        for index, item in enumerate(inventory):
            path = Path(item['path'])
            try:
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                row = measure_file(path)
            except (OSError, ValueError, RuntimeError) as error:
                row = {'path': str(path), 'error_type': type(error).__name__}
            records.append(row)
            output.write(json.dumps(row, allow_nan=False) + '\n')
            if (index + 1) % 1000 == 0:
                print(f'Measured {index + 1}/{len(inventory)} files', flush=True)
    summary = summarize(records)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'attempted_files': summary['attempted_files'], 'decoded_files': summary['decoded_files'],
                      'decode_errors': summary['decode_errors']}))


if __name__ == '__main__':
    main()
