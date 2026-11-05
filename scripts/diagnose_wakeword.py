"""
Diagnóstico del pipeline wake word → audio.
Graba 4s del ReSpeaker, analiza los 6 canales, y corre hey_jarvis con ambos
formatos (float32 normalizado vs int16-scale) para encontrar la falla exacta.

Uso: systemctl --user stop kza-voice.service  # liberar el mic
     .venv/bin/python scripts/diagnose_wakeword.py
     systemctl --user start kza-voice.service
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from src.wakeword.live_capture import LiveWakewordDiagnosticsCapture

SAMPLE_RATE = 16000
DURATION = 4.0
DEVICE = 9
CHUNK_SAMPLES = 1280  # 80ms @ 16kHz — tamaño esperado por OpenWakeWord


def analyze_channels(audio_6ch: np.ndarray, *, emit: Callable[..., None] = print) -> None:
    emit("\n=== Actividad por canal (RMS, peak) ===")
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
        emit(f"  ch{ch} ({label:14}): RMS={rms:.4f}  peak={peak:.4f}{marker}")


def run_openwakeword(audio_mono_f32: np.ndarray, scale_label: str, *,
                     capture: LiveWakewordDiagnosticsCapture | None = None) -> None:
    """Report raw model scores, fencing each live call and output when bound."""
    if capture is not None:
        from src.wakeword.live_capture import LiveWakewordDiagnosticsCapture
        if not isinstance(capture, LiveWakewordDiagnosticsCapture):
            raise RuntimeError("wakeword_capture_authority_required")

    def require_current() -> None:
        if capture is not None:
            capture.require_current()

    def emit(*values, **kwargs) -> None:
        if capture is None:
            print(*values, **kwargs)
        else:
            capture.publish(print, *values, **kwargs)

    oww = preds = chunk = score = top = s = None
    scores: list[float] = []
    failure = None
    try:
        require_current()
        from openwakeword import Model
        emit(f"\n=== Predict con '{scale_label}' ===")
        def factory():
            return Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        oww = factory() if capture is None else capture.own_python_model(factory)
        for start in range(0, len(audio_mono_f32) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
            require_current()
            chunk = audio_mono_f32[start:start + CHUNK_SAMPLES]
            preds = oww.predict(chunk)
            require_current()
            score = preds.get("hey_jarvis", 0.0)
            scores.append(score)
        if scores:
            emit(f"  chunks: {len(scores)}  max: {max(scores):.4f}  mean: {np.mean(scores):.4f}")
            top = sorted(enumerate(scores), key=lambda x: -x[1])[:5]
            for idx, s in top:
                t_ms = int(idx * CHUNK_SAMPLES / SAMPLE_RATE * 1000)
                emit(f"    t={t_ms}ms  score={s:.4f}")
        else:
            emit("  (sin chunks — audio demasiado corto)")
    except BaseException as error:
        if capture is None:
            raise
        failure = capture.model_failure(error)
    finally:
        # Drop local references before the owner's release and retire receipt.
        oww = preds = chunk = score = top = s = None
        scores.clear()
        if capture is not None:
            capture.release_python_model()
    if failure is not None:
        raise failure


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico de captura y modelos wakeword")
    parser.add_argument("--dashboard-url", required=True,
                        help="Origen del Dashboard que controla la captura")
    parser.add_argument("--token-env", default="KZA_DASHBOARD_TOKEN",
                        help="Variable de entorno con la credencial administrativa")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if not token:
        parser.error("Se requiere una credencial administrativa en la variable indicada")
    from src.audio.native_capture_authority import WakewordDiagnosticsAuthorityClient
    from src.wakeword.live_capture import LiveWakewordDiagnosticsCapture

    authority = WakewordDiagnosticsAuthorityClient(args.dashboard_url, token)
    with LiveWakewordDiagnosticsCapture(authority) as live:
        import sounddevice as sd

        def emit(*values, **kwargs) -> None:
            live.publish(print, *values, **kwargs)

        buffers: list[np.ndarray] = []
        stream = block = None
        try:
            emit("=== DIAGNÓSTICO WAKE WORD ===")
            emit(f"Device {DEVICE}, {DURATION}s, {SAMPLE_RATE} Hz, 6 canales")
            emit(f"\nEn 3 segundos empieza la grabación. Decí 'hey jarvis' 2-3 veces durante los {DURATION}s.\n")
            for c in (3, 2, 1):
                emit(f"  {c}...", end="", flush=True)
                live.pause(1)
            emit(" 🎙️  ¡HABLA!")
            samples = int(SAMPLE_RATE * DURATION)
            audio_6ch = np.empty((samples, 6), dtype=np.float32)
            buffers.append(audio_6ch)
            stream = live.open_stream(lambda: sd.InputStream(
                samplerate=SAMPLE_RATE, channels=6, device=DEVICE,
                dtype="float32", blocksize=CHUNK_SAMPLES,
            ))
            offset = 0
            while offset < samples:
                live.require_current()
                frames = min(CHUNK_SAMPLES, samples - offset)
                block, _overflowed = stream.read(frames)
                live.require_current()
                if block.shape != (frames, 6):
                    raise RuntimeError("wakeword_diagnostic_invalid_frames")
                audio_6ch[offset:offset + frames] = block
                offset += frames
            live.close_stream()
            stream = block = None
            emit(f"\nGrabados {audio_6ch.shape[0]} samples × {audio_6ch.shape[1]} canales.")
            analyze_channels(audio_6ch, emit=emit)
            live.require_current()
            rms_by_ch = [float(np.sqrt(np.mean(audio_6ch[:, c] ** 2))) for c in range(6)]
            best_ch = int(np.argmax(rms_by_ch))
            emit(f"\nCanal más activo: ch{best_ch} (RMS={rms_by_ch[best_ch]:.4f})")
            ch0 = audio_6ch[:, 0].copy()
            buffers.append(ch0)
            run_openwakeword(ch0, "ch0, float32 normalizado [-1,1]", capture=live)
            live.require_current()
            ch0_i16 = (np.clip(ch0, -1.0, 1.0) * 32767).astype(np.float32)
            buffers.append(ch0_i16)
            run_openwakeword(ch0_i16, "ch0, float32 escala int16 [-32767,32767]", capture=live)
            if best_ch != 0:
                live.require_current()
                chN = audio_6ch[:, best_ch].copy()
                buffers.append(chN)
                chN_i16 = (np.clip(chN, -1.0, 1.0) * 32767).astype(np.float32)
                buffers.append(chN_i16)
                run_openwakeword(chN_i16, f"ch{best_ch}, escala int16", capture=live)
            emit("\n=== Interpretación ===")
            emit("- Si max score >= 0.3 en algún canal/formato → ese es el correcto, fix el código del detector.")
            emit("- Si max score < 0.1 en TODOS los formatos → el modelo inglés no pilla tu pronunciación. Hacer training custom.")
            emit("- Si ch0 está silencioso pero otro ch tiene RMS alto → el firmware del ReSpeaker usa otro canal para audio procesado.")
        finally:
            stream = block = None
            for buffer in buffers:
                buffer.fill(0)
            buffers.clear()


if __name__ == "__main__":
    main()
