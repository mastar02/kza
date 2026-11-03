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
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.resource_retirement import (  # noqa: E402
    ResourceFailure,
    ResourceOwner,
    ResourceSession,
    raise_errors,
)

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
        audio = None
        try:
            audio = load_wav_mono16k(path)
            durations_s.append(len(audio) / TARGET_SR)
            text, ms = transcribe_fn(audio)
            latencies.append(ms)
            if not text.strip():
                empties += 1
            if i < 8:
                samples.append((path.name, text[:60]))
        finally:
            audio = None
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


class _BenchmarkEngine(ResourceOwner):
    """Keep STT initialization/conversion separate from shared retirement."""

    def __init__(self, stt, *, owns_stt: bool = False):
        if type(owns_stt) is not bool:
            raise ValueError("benchmark_ownership_must_be_bool")
        super().__init__(stt, owns_resource=owns_stt, error_prefix="benchmark",
                         thread_name="benchmark-stt-close")

    @property
    def stt(self):
        return self.resource

    @property
    def owns_stt(self) -> bool:
        return self.owns_resource

    def initialize(self) -> None:
        """Admit load and warmup as one operation, including SDK callbacks."""
        failure = None
        with self.operation():
            stt = self.stt
            try:
                stt.load()
                self(np.zeros(TARGET_SR, dtype=np.float32))
            except BaseException as error:
                failure = ResourceFailure.capture("benchmark_initialize_failed", error)
            finally:
                stt = None
        if failure is not None:
            raise failure.exception()

    def __call__(self, audio):
        failure = None
        result = stt = r = None
        try:
            with self.operation():
                stt = self.stt
                try:
                    r = stt.transcribe_with_confidence(audio)
                    result = r.text, r.elapsed_ms
                except BaseException as error:
                    failure = ResourceFailure.capture("benchmark_transcription_failed", error)
                finally:
                    # Discard loans before publishing admitted-operation exit.
                    stt = r = audio = None
            if failure is not None:
                raise failure.exception()
            return result
        finally:
            # Admission may reject before the inner finally has been entered.
            # A retained rejection traceback must not retain the caller's PCM.
            stt = r = audio = None


def _tc(stt, owns_stt: bool = False) -> _BenchmarkEngine:
    """Explicit callable (text, ms) adapter; external instances are borrowed."""
    return _BenchmarkEngine(stt, owns_stt=owns_stt)


def _initialize_engine(
    engine: _BenchmarkEngine,
    register: Callable[[_BenchmarkEngine], None] | None = None,
) -> _BenchmarkEngine:
    registered = False
    try:
        if register is not None:
            register(engine)
            registered = True
        engine.initialize()
    except BaseException as prior_error:
        try:
            if registered:
                # The enclosing session must start every other engine's
                # retirement before joining this potentially held native exit.
                engine.begin_close()
            else:
                engine.close()
        except BaseException as error:
            raise_errors([prior_error, error], message="benchmark_retirement_failed")
        raise
    return engine

def build_parakeet(
    model_name: str, language: str, *,
    register: Callable[[_BenchmarkEngine], None] | None = None,
) -> _BenchmarkEngine:
    """Build and warm an owned Parakeet benchmark callable.

    Args:
        model_name: Existing ONNX ASR model selection.
        language: Transcription language.
        register: Receives the owner before load and owns cleanup on failure.

    Returns:
        Callable whose close waits for the adapter's actual retirement.
    """
    from src.stt.parakeet_stt import ParakeetSTT

    stt = ParakeetSTT(model_name=model_name, language=language)
    engine = _tc(stt, owns_stt=True)
    stt = None
    return _initialize_engine(engine, register)


def build_whisper(
    model: str, device: str, compute_type: str, language: str, *,
    register: Callable[[_BenchmarkEngine], None] | None = None,
) -> _BenchmarkEngine:
    """Build and warm an owned Whisper benchmark callable.

    Args:
        model: Existing Whisper model path or identifier.
        device: Requested device without changing application allocation.
        compute_type: Existing inference precision.
        language: Transcription language.
        register: Receives the owner before load and owns cleanup on failure.

    Returns:
        Callable whose close waits for the adapter and its native worker.
    """
    from src.stt.whisper_fast import FastWhisperSTT

    stt = FastWhisperSTT(
        model=model, device=device, compute_type=compute_type,
        language=language, beam_size=1, vad_filter=False,
    )
    engine = _tc(stt, owns_stt=True)
    stt = None
    return _initialize_engine(engine, register)


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

    with ResourceSession(error_prefix="benchmark") as engines:
        print("\nCargando Parakeet…")
        pk = build_parakeet(args.parakeet_model, args.language, register=engines.register)
        if captured:
            _print_report("COMANDOS (captured)", _run_engine("parakeet", pk, captured))
        if rejected:
            _print_report("RUIDO/TV (rejected)", _run_engine("parakeet", pk, rejected))

        if args.with_whisper:
            print("\nCargando Whisper…")
            wh = build_whisper(args.whisper_model, args.whisper_device,
                               args.whisper_compute, args.language, register=engines.register)
            if captured:
                _print_report("COMANDOS (captured)", _run_engine("whisper", wh, captured))
            if rejected:
                _print_report("RUIDO/TV (rejected)", _run_engine("whisper", wh, rejected))

    print("\nGate de decisión (plan): Parakeet p95 < ~200ms sobre 'captured' Y "
          "vacío(rejected) claramente > Whisper → seguir a shadow en vivo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
