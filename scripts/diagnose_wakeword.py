"""
Diagnóstico del pipeline wake word → audio.
Graba 4s del ReSpeaker, analiza los 6 canales, y corre hey_jarvis con ambos
formatos (float32 normalizado vs int16-scale) para encontrar la falla exacta.

Uso: systemctl --user stop kza-voice.service  # liberar el mic
     .venv/bin/python scripts/diagnose_wakeword.py
     systemctl --user start kza-voice.service
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_RATE = 16000
DURATION = 4.0
DEVICE = 9
CHUNK_SAMPLES = 1280  # 80ms @ 16kHz — tamaño esperado por OpenWakeWord


def analyze_channels(audio_6ch: np.ndarray) -> None:
    print("\n=== Actividad por canal (RMS, peak) ===")
    for ch in range(audio_6ch.shape[1]):
        data = audio_6ch[:, ch]
        rms = float(np.sqrt(np.mean(data ** 2)))
        peak = float(np.max(np.abs(data)))
        label = {
            0: "procesado/AEC",
            1: "playback-ref",
            2: "mic raw 0",
            3: "mic raw 1",
            4: "mic raw 2",
            5: "mic raw 3",
        }.get(ch, "?")
        marker = " ← ACTIVO" if rms > 0.005 else " (silencio)"
        print(f"  ch{ch} ({label:14}): RMS={rms:.4f}  peak={peak:.4f}{marker}")


def run_openwakeword(audio_mono_f32: np.ndarray, scale_label: str) -> None:
    """Pasa el audio por OpenWakeWord en chunks y reporta max score."""
    from openwakeword import Model
    print(f"\n=== Predict con '{scale_label}' ===")
    oww = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

    scores: list[float] = []
    # Feed en chunks de 80ms
    for start in range(0, len(audio_mono_f32) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
        chunk = audio_mono_f32[start:start + CHUNK_SAMPLES]
        preds = oww.predict(chunk)
        score = preds.get("hey_jarvis", 0.0)
        scores.append(score)

    if scores:
        print(f"  chunks: {len(scores)}  max: {max(scores):.4f}  mean: {np.mean(scores):.4f}")
        # Top 5 highest
        top = sorted(enumerate(scores), key=lambda x: -x[1])[:5]
        for idx, s in top:
            t_ms = int(idx * CHUNK_SAMPLES / SAMPLE_RATE * 1000)
            print(f"    t={t_ms}ms  score={s:.4f}")
    else:
        print("  (sin chunks — audio demasiado corto)")


def main():
    print(f"=== DIAGNÓSTICO WAKE WORD ===")
    print(f"Device {DEVICE}, {DURATION}s, {SAMPLE_RATE} Hz, 6 canales")
    print(f"\nEn 3 segundos empieza la grabación. Decí 'hey jarvis' 2-3 veces durante los {DURATION}s.\n")
    import time
    for c in (3, 2, 1):
        print(f"  {c}...", end="", flush=True)
        time.sleep(1)
    print(" 🎙️  ¡HABLA!")

    samples = int(SAMPLE_RATE * DURATION)
    audio_6ch = sd.rec(samples, samplerate=SAMPLE_RATE, channels=6,
                        device=DEVICE, dtype="float32")
    sd.wait()
    print(f"\nGrabados {audio_6ch.shape[0]} samples × {audio_6ch.shape[1]} canales.")

    analyze_channels(audio_6ch)

    # Identificar el canal más activo (probable candidato "procesado")
    rms_by_ch = [float(np.sqrt(np.mean(audio_6ch[:, c] ** 2))) for c in range(6)]
    best_ch = int(np.argmax(rms_by_ch))
    print(f"\nCanal más activo: ch{best_ch} (RMS={rms_by_ch[best_ch]:.4f})")

    # 1) Prueba con ch0 float32 normalizado [-1,1]
    ch0 = audio_6ch[:, 0].copy()
    run_openwakeword(ch0, "ch0, float32 normalizado [-1,1]")

    # 2) Prueba con ch0 escalado a int16-range
    ch0_i16 = (np.clip(ch0, -1.0, 1.0) * 32767).astype(np.float32)
    run_openwakeword(ch0_i16, "ch0, float32 escala int16 [-32767,32767]")

    # 3) Prueba con el canal más activo (si es distinto de ch0)
    if best_ch != 0:
        chN = audio_6ch[:, best_ch].copy()
        chN_i16 = (np.clip(chN, -1.0, 1.0) * 32767).astype(np.float32)
        run_openwakeword(chN_i16, f"ch{best_ch}, escala int16")

    print("\n=== Interpretación ===")
    print("- Si max score >= 0.3 en algún canal/formato → ese es el correcto, fix el código del detector.")
    print("- Si max score < 0.1 en TODOS los formatos → el modelo inglés no pilla tu pronunciación. Hacer training custom.")
    print("- Si ch0 está silencioso pero otro ch tiene RMS alto → el firmware del ReSpeaker usa otro canal para audio procesado.")


if __name__ == "__main__":
    main()
