# tools/make_earcon.py
"""Genera el earcon 'no entendí' (dos tonos descendentes, ~200ms, 24kHz mono).

24kHz float32 = formato del ResponseCache (Kokoro). Reproducible: no se
commitea el WAV binario, se regenera con `python -m tools.make_earcon`.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 24000
OUT = Path("data/earcons/not_understood.wav")


def _tone(freq: float, ms: int, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, ms / 1000.0, int(sr * ms / 1000.0), endpoint=False)
    wave_ = 0.4 * np.sin(2 * np.pi * freq * t)
    # fade in/out 8ms para evitar clicks
    fade = int(sr * 0.008)
    env = np.ones_like(wave_)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (wave_ * env).astype(np.float32)


def main() -> None:
    # G5 → C5 descendente = "uh-oh" sutil, no alarmante.
    sig = np.concatenate([_tone(784.0, 90), _tone(523.0, 110)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(sig, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype(np.int16)
    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm16.tobytes())
    print(f"earcon escrito: {OUT} ({len(sig)/SR*1000:.0f}ms)")


if __name__ == "__main__":
    main()
