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
import asyncio
import logging
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import Context
from dataclasses import dataclass
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


_ENGINES: set = set()


_FAILED_ENGINES: set = set()


_ENGINES_LOCK = threading.Lock()


@dataclass(frozen=True)
class _Failure:
    """Keep classification, never a native exception/traceback or payload."""

    stage: str
    kind: str
    exit_code: int | None = None

    @classmethod
    def capture(cls, stage: str, error: BaseException):
        if isinstance(error, KeyboardInterrupt):
            return cls(stage, "interrupt")
        if isinstance(error, SystemExit):
            code = error.code
            return cls(stage, "exit", code if code is None or type(code) in (int, bool) else 1)
        if isinstance(error, GeneratorExit):
            return cls(stage, "generator")
        if isinstance(error, asyncio.CancelledError):
            return cls(stage, "cancel")
        if not isinstance(error, Exception):
            return cls(stage, "control")
        return cls(stage, type(error).__name__)

    def exception(self) -> BaseException:
        if self.kind == "interrupt":
            return KeyboardInterrupt()
        if self.kind == "exit":
            return SystemExit(self.exit_code)
        if self.kind == "generator":
            return GeneratorExit()
        if self.kind == "cancel":
            return asyncio.CancelledError()
        if self.kind == "control":
            return BaseException(f"{self.stage}:control")
        return RuntimeError(f"{self.stage}:{self.kind}")


def _raise_errors(errors: list[BaseException]) -> None:
    for error in errors:
        if not isinstance(error, Exception):
            raise error
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup("benchmark_retirement_failed", errors)


class _StartAdmission:
    """Identify the sole native-start transaction and its originating caller."""

    def __init__(self, owner: threading.Thread):
        self.owner = owner
        self.finished = False

class _BenchmarkEngine:
    """Explicit callable owner; borrowed adapters never receive close or seal."""

    def __init__(self, stt, *, owns_stt: bool = False):
        if type(owns_stt) is not bool:
            raise ValueError("benchmark_ownership_must_be_bool")
        self.stt, self.owns_stt = stt, owns_stt
        self._condition = threading.Condition()
        self._sealed = False
        self._admitted = 0
        self._workers: dict[threading.Thread, int] = {}
        self._start_called = False
        self._start_admission: _StartAdmission | None = None
        self._admission_done = threading.Event()
        self._target_exited = threading.Event()
        self._start_failure = self._close_failure = None
        self._confirmed = False
        self._closer = threading.Thread(
            target=Context().run, args=(self._retire,),
            name="benchmark-stt-close", daemon=True,
        )
        self._handle = getattr(self._closer, "_handle", None)
        handle_type = getattr(threading, "_ThreadHandle", None)
        started = getattr(self._closer, "_started", None)
        # Audit boundary shared with the local K/D native-thread receipts; no
        # STT private helper import, and no actor starts before this guard.
        if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13)
                or handle_type is None or type(self._handle) is not handle_type
                or type(getattr(self._handle, "ident", None)) is not int or self._handle.ident != 0
                or not callable(getattr(self._handle, "join", None))
                or not callable(getattr(self._handle, "is_done", None))
                or not callable(getattr(started, "is_set", None)) or started.is_set()):
            raise RuntimeError("benchmark_close_unsupported_runtime")
        if owns_stt:
            # The adapter is explicit and retained before load or warmup.
            with _ENGINES_LOCK:
                _ENGINES.add(self)

    @contextmanager
    def _operation(self):
        caller = threading.current_thread()
        with self._condition:
            if self._sealed and not self._workers.get(caller, 0):
                raise RuntimeError("benchmark_engine_closed")
            self._admitted += 1
            self._workers[caller] = self._workers.get(caller, 0) + 1
        try:
            yield
        finally:
            with self._condition:
                self._admitted -= 1
                remaining = self._workers[caller] - 1
                if remaining:
                    self._workers[caller] = remaining
                else:
                    self._workers.pop(caller)
                self._condition.notify_all()

    def initialize(self) -> None:
        """Admit load and warmup as one operation, including SDK callbacks."""
        failure = None
        with self._operation():
            stt = self.stt
            try:
                stt.load()
                self(np.zeros(TARGET_SR, dtype=np.float32))
            except BaseException as error:
                failure = _Failure.capture("benchmark_initialize_failed", error)
            finally:
                stt = None
        if failure is not None:
            raise failure.exception()

    def __call__(self, audio):
        failure = None
        result = stt = r = None
        try:
            with self._operation():
                stt = self.stt
                try:
                    r = stt.transcribe_with_confidence(audio)
                    result = r.text, r.elapsed_ms
                except BaseException as error:
                    failure = _Failure.capture("benchmark_transcription_failed", error)
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

    def require_close_caller(self) -> None:
        caller = threading.current_thread()
        with self._condition:
            if self._workers.get(caller, 0):
                raise RuntimeError("benchmark_close_from_operation")
            if caller is self._closer:
                raise RuntimeError("benchmark_close_from_owner")
            admission = self._start_admission
            if admission is not None and admission.owner is caller and not admission.finished:
                raise RuntimeError("benchmark_close_from_admission")
            stt = self.stt
        # FastWhisper dispatches a caller's load/transcribe onto its own pinned
        # Thread. Check that actual Thread before creating a fresh-context
        # closer, which would otherwise hide its close-from-worker origin.
        # These are adapter metadata only; no model or private helper is called.
        try:
            worker = vars(stt).get("_worker_thread") if stt is not None else None
        except TypeError:
            worker = None
        if caller is worker:
            raise RuntimeError("benchmark_close_from_operation")

    def request_close(self) -> None:
        """Seal only this callable; already admitted load/warmup may finish."""
        with self._condition:
            self._sealed = True
            self._condition.notify_all()

    def begin_close(self) -> threading.Thread:
        """Resolve one native-start admission even if pre-dispatch metadata fails."""
        self.require_close_caller()
        self.request_close()
        admission = None
        try:
            with self._condition:
                if self._start_admission is None:
                    admission = _StartAdmission(threading.current_thread())
                    # Owner identity is part of the single published token.
                    # A follower never acquires responsibility for this start.
                    self._start_admission = admission
                    self._start_called = True
            if admission is not None:
                with _ENGINES_LOCK:
                    _ENGINES.add(self)
                # No lifecycle lock across SDK/native thread admission.
                self._closer.start()
        except BaseException as error:
            if admission is None or self._start_admission is not admission:
                raise
            self._start_failure = _Failure.capture("benchmark_close_start_failed", error)
        finally:
            if admission is not None and self._start_admission is admission:
                try:
                    self._admission_done.set()
                finally:
                    # This plain receipt also covers notification interruption.
                    # It does not claim that Thread.start dispatched or exited.
                    admission.finished = True
        return self._closer

    def _retire(self) -> None:
        stt = None
        try:
            with self._condition:
                while self._admitted:
                    self._condition.wait()
                stt = self.stt
            if self.owns_stt:
                # Existing model closure supplies its real native/executor
                # retirement. No backend/VRAM disposal method is invented.
                stt.close()
        except BaseException as error:
            self._close_failure = _Failure.capture("benchmark_engine_close_failed", error)
        finally:
            stt = None
            self._target_exited.set()

    def wait_closed(self) -> None:
        """Observe actual native exit through repeated observer interruption."""
        self.require_close_caller()
        closer = self.begin_close()
        controls = []
        failure = None
        admission = self._start_admission
        while admission is not None and not admission.finished:
            try:
                # Notification is advisory: its publisher may be interrupted.
                # Never wait indefinitely on an unset Event after resolution.
                self._admission_done.wait(.05)
                if not admission.finished and self._admission_done.is_set():
                    time.sleep(0)
            except BaseException as error:
                item = _Failure.capture("benchmark_close_wait_failed", error)
                if isinstance(error, Exception):
                    failure = item
                    break
                controls.append(item)
        if failure is None:
            if self._handle.ident == 0:
                failure = _Failure("benchmark_close_unconfirmed", "start")
            else:
                while True:
                    try:
                        if not closer._started.is_set():
                            self._handle.join()
                        else:
                            closer.join()
                        if not self._handle.is_done() or not self._target_exited.is_set():
                            failure = _Failure("benchmark_close_unconfirmed", "exit")
                        break
                    except BaseException as error:
                        item = _Failure.capture("benchmark_close_wait_failed", error)
                        if isinstance(error, Exception):
                            failure = item
                            break
                        controls.append(item)
        if failure is None and self._close_failure is None:
            with self._condition:
                self.stt = None
                self._confirmed = True
            with _ENGINES_LOCK:
                _ENGINES.discard(self)
        else:
            with _ENGINES_LOCK:
                _FAILED_ENGINES.add(self)
        errors = [item.exception() for item in
                  (self._start_failure, self._close_failure, failure, *controls) if item is not None]
        _raise_errors(errors)

    def close(self) -> None:
        """Retire this callable; only an owning handle retires its adapter."""
        self.wait_closed()


class _BenchmarkSession:
    """Own only the distinct engines constructed for this script invocation."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._engines: dict[int, _BenchmarkEngine] = {}
        self._sealed = False

    def register(self, engine: _BenchmarkEngine) -> None:
        with self._condition:
            if self._sealed:
                raise RuntimeError("benchmark_session_closed")
            self._engines[id(engine)] = engine

    def __enter__(self):
        return self

    def __exit__(self, _kind, prior_error, _traceback):
        # Every guard is metadata-only, before any seal or context-empty child.
        with self._condition:
            engines = tuple(self._engines.values())
            for engine in engines:
                engine.require_close_caller()
            self._sealed = True
        errors = []
        for engine in engines:
            try:
                engine.request_close()
            except BaseException as error:
                errors.append(error)
        for engine in engines:
            try:
                engine.begin_close()
            except BaseException as error:
                errors.append(error)
        for engine in engines:
            try:
                engine.wait_closed()
            except BaseException as error:
                errors.append(error)
        if errors:
            _raise_errors(([prior_error] if prior_error is not None else []) + errors)
        return False

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
            _raise_errors([prior_error, error])
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

    with _BenchmarkSession() as engines:
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
