#!/usr/bin/env python3
"""
Benchmark A/B offline: Parakeet vs Whisper sobre audio real del XVF3800.

Responde "¿es viable Parakeet en el fast path?" SIN tocar el servicio vivo.
Corre en el server (tiene el modelo Parakeet cacheado + los wavs). Mide:
  - Latencia real de Parakeet (p50/p95/max) sobre clips de comando reales.
  - Tasa de texto vacío sobre 'rejected/' (proxy anti-alucinación: Parakeet
    debería callar sobre ruido/TV donde Whisper alucinaba texto coherente).
  - Con --with-whisper: mismas métricas para Whisper + diffs de transcripción.

Datos (en el server): data/wakeword_training/captured/ (comandos post-wake) y
captured/rejected/ (falsos positivos / ruido / TV).

Uso:
    python tools/benchmark_stt_ab.py --limit 300
    python tools/benchmark_stt_ab.py --limit 300 --with-whisper --whisper-device cpu
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_CAPTURED = "data/wakeword_training/captured"
DEFAULT_REJECTED = "data/wakeword_training/captured/rejected"
TARGET_SR = 16000


def load_wav_mono16k(path: Path) -> np.ndarray:
    """Leer un wav a float32 mono 16 kHz (resample naive si hiciera falta)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        # resample lineal simple; el corpus ya es 16k, esto es defensa
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * TARGET_SR / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return audio


def _wavs(directory: str, limit: int) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        return []
    files = sorted(p for p in d.glob("*.wav"))
    return files[:limit] if limit else files


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _run_engine(name: str, transcribe_fn, wavs: list[Path]) -> dict:
    """Transcribir todos los wavs con transcribe_fn(audio)->(text, ms)."""
    latencies: list[float] = []
    empties = 0
    durations_s: list[float] = []
    samples: list[tuple[str, str]] = []
    for i, path in enumerate(wavs):
        audio = load_wav_mono16k(path)
        durations_s.append(len(audio) / TARGET_SR)
        text, ms = transcribe_fn(audio)
        latencies.append(ms)
        if not text.strip():
            empties += 1
        if i < 8:
            samples.append((path.name, text[:60]))
    total_audio = sum(durations_s) or 1e-9
    rtf = (sum(latencies) / 1000.0) / total_audio
    return {
        "name": name,
        "n": len(wavs),
        "p50": _pct(latencies, 50),
        "p95": _pct(latencies, 95),
        "max": max(latencies) if latencies else 0.0,
        "empty_pct": 100.0 * empties / len(wavs) if wavs else 0.0,
        "rtf": rtf,
        "samples": samples,
    }


def _print_report(title: str, res: dict) -> None:
    print(f"\n── {title} · {res['name']} (n={res['n']}) ──")
    print(f"  latencia  p50={res['p50']:.0f}ms  p95={res['p95']:.0f}ms  "
          f"max={res['max']:.0f}ms  RTF={res['rtf']:.3f}")
    print(f"  vacío     {res['empty_pct']:.1f}%")
    for fname, txt in res["samples"]:
        print(f"    {fname}: {txt!r}")


def _tc(stt):
    """Adaptador a (text, ms) vía transcribe_with_confidence.

    Lo usan todos los motores (existe en Parakeet y Whisper), así el script
    corre tanto con el árbol refactorizado como con el de producción.
    """
    def fn(audio):
        r = stt.transcribe_with_confidence(audio)
        return r.text, r.elapsed_ms
    return fn


def build_parakeet(model_name: str, language: str):
    try:  # tras el refactor 2026-07
        from src.stt.parakeet_stt import ParakeetSTT
    except ImportError:  # árbol de producción (aún en ambient)
        from src.ambient.parakeet_stt import ParakeetSTT

    stt = ParakeetSTT(model_name=model_name, language=language)
    stt.load()
    fn = _tc(stt)
    fn(np.zeros(TARGET_SR, dtype=np.float32))  # warm-up (init onnxruntime)
    return fn


def build_whisper(model: str, device: str, compute_type: str, language: str):
    from src.stt.whisper_fast import FastWhisperSTT

    stt = FastWhisperSTT(
        model=model, device=device, compute_type=compute_type,
        language=language, beam_size=1, vad_filter=False,
    )
    stt.load()
    fn = _tc(stt)
    fn(np.zeros(TARGET_SR, dtype=np.float32))  # warm-up
    return fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captured-dir", default=DEFAULT_CAPTURED)
    ap.add_argument("--rejected-dir", default=DEFAULT_REJECTED)
    ap.add_argument("--limit", type=int, default=300,
                    help="máx wavs por carpeta (0 = todos)")
    ap.add_argument("--language", default="es")
    ap.add_argument("--parakeet-model", default="nemo-parakeet-tdt-0.6b-v3")
    ap.add_argument("--with-whisper", action="store_true",
                    help="correr también Whisper para comparar texto")
    ap.add_argument("--whisper-model", default="./models/whisper-v3-turbo")
    ap.add_argument("--whisper-device", default="cpu",
                    help="cpu (no compite con GPU de prod) | cuda:1")
    ap.add_argument("--whisper-compute", default="int8")
    args = ap.parse_args()

    captured = _wavs(args.captured_dir, args.limit)
    rejected = _wavs(args.rejected_dir, args.limit)
    print(f"captured={len(captured)} wavs · rejected={len(rejected)} wavs")
    if not captured and not rejected:
        print("ERROR: no se encontraron wavs. ¿Correr desde la raíz del repo?")
        return 1

    print("\nCargando Parakeet…")
    pk = build_parakeet(args.parakeet_model, args.language)
    if captured:
        _print_report("COMANDOS (captured)", _run_engine("parakeet", pk, captured))
    if rejected:
        _print_report("RUIDO/TV (rejected)", _run_engine("parakeet", pk, rejected))

    if args.with_whisper:
        print("\nCargando Whisper…")
        wh = build_whisper(args.whisper_model, args.whisper_device,
                           args.whisper_compute, args.language)
        if captured:
            _print_report("COMANDOS (captured)", _run_engine("whisper", wh, captured))
        if rejected:
            _print_report("RUIDO/TV (rejected)", _run_engine("whisper", wh, rejected))

    print("\nGate de decisión (plan): Parakeet p95 < ~200ms sobre 'captured' Y "
          "vacío(rejected) claramente > Whisper → seguir a shadow en vivo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
