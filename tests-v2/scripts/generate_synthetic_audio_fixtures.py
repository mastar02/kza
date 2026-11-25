#!/usr/bin/env python3
"""Generate deterministic, zero-PII 16kHz mono synthetic WAV fixtures for L3 Golden Audio Replay.

Every clip is pure deterministic DSP math (fixed PRNG seed, zero human recordings or PII),
keeping total fixture size tiny (<500 KB) and byte-reproducible across machines.
"""

from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 16_000
OUT_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "audio"


def _write_wav(path: Path, samples: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = bytearray()
    for s in samples:
        clamped = max(-1.0, min(1.0, s))
        pcm.extend(struct.pack("<h", int(round(clamped * 32767.0))))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(pcm))


def _speech_like_burst(duration_s: float, amplitude: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = int(SAMPLE_RATE * duration_s)
    out: list[float] = []
    for i in range(n):
        t = i / SAMPLE_RATE
        # Syllable-rate amplitude envelope (~4 Hz) with leading/trailing silence pads.
        if t < 0.25 or t > duration_s - 0.35:
            env = 0.0
        else:
            env = 0.5 * (1.0 + math.sin(2.0 * math.pi * 4.0 * (t - 0.25) - math.pi / 2.0))
        # Multi-formant harmonic stack (F0=150Hz + 450Hz + 900Hz) + slight aspiration noise.
        sig = (
            0.55 * math.sin(2.0 * math.pi * 150.0 * t)
            + 0.30 * math.sin(2.0 * math.pi * 450.0 * t)
            + 0.15 * math.sin(2.0 * math.pi * 900.0 * t)
            + 0.05 * rng.uniform(-1.0, 1.0)
        )
        out.append(amplitude * env * sig)
    return out


def main() -> int:
    rng = random.Random(20261125)

    # 1. Pure silence (2.0s) — must never trigger VAD or wake.
    _write_wav(OUT_DIR / "silence_2s.wav", [0.0] * (SAMPLE_RATE * 2))

    # 2. Steady low-level broadband fan/room noise (2.0s, RMS ~0.015).
    _write_wav(
        OUT_DIR / "room_noise_2s.wav",
        [rng.gauss(0.0, 0.015) for _ in range(SAMPLE_RATE * 2)],
    )

    # 3. Clean near-field speech-band command envelope (2.0s, peak ~0.45).
    _write_wav(OUT_DIR / "near_field_burst_2s.wav", _speech_like_burst(2.0, 0.45, seed=101))

    # 4. Far-field low-RMS speech-band envelope (2.0s, peak ~0.045) + noise floor.
    far = _speech_like_burst(2.0, 0.045, seed=202)
    _write_wav(
        OUT_DIR / "far_field_low_rms_2s.wav",
        [s + rng.gauss(0.0, 0.008) for s in far],
    )

    # 5. Continuous TV/background chatter hard-negative envelope (3.0s, no clean pause).
    tv = [
        0.18 * (0.6 + 0.4 * math.sin(2.0 * math.pi * 2.3 * (i / SAMPLE_RATE)))
        * (
            0.5 * math.sin(2.0 * math.pi * 220.0 * (i / SAMPLE_RATE))
            + 0.3 * math.sin(2.0 * math.pi * 680.0 * (i / SAMPLE_RATE))
            + 0.2 * rng.uniform(-1.0, 1.0)
        )
        for i in range(SAMPLE_RATE * 3)
    ]
    _write_wav(OUT_DIR / "tv_chatter_hard_neg_3s.wav", tv)

    # 6. Sharp mid-TTS barge-in transient (1.2s).
    barge = _speech_like_burst(1.2, 0.60, seed=303)
    _write_wav(OUT_DIR / "barge_in_burst_1_2s.wav", barge)

    print(f"Generated 6 synthetic 16kHz mono WAV fixtures in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
