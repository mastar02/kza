"""Harness de calibración acústica — matriz voz/TV/silencio (spec 2026-06-05).

Mide simultáneamente, bajo una condición etiquetada (silencio/tv/voz/voz_tv):
  - RMS por chunk del stream del mic (mismo device/canal que prod)
  - score máximo de openwakeword (nexa.onnx) por chunk
  - SPENERGY[3] del XVF3800 (poll 25Hz vía XvfController, fail-open)

La pregunta que responde: ¿qué señal tiene gap entre voz(p5) y tv(p95)?
Esa señal es compuerta viable; las que no separan quedan documentadas como
muertas con el chip en su estado actual (MAXGAIN=8, ch1 ASR).

La captura requiere un Dashboard disponible y una credencial administrativa.
El dispositivo debe estar cedido a esta invocación para evitar contención del
micrófono y USB. No detener la autoridad que confirma el retiro de la captura.

Uso (entorno de desarrollo y credencial en una variable de entorno):
    python -m tools.acoustic_calibration --condition silencio --duration 120 \
        --device 4 --channel 1 --model models/wakeword/nexa.onnx \
        --dashboard-url http://127.0.0.1:9500 --token-env KZA_CALIBRATION_TOKEN
    python -m tools.acoustic_calibration --condition tv --duration 180 ...
    python -m tools.acoustic_calibration --condition voz --duration 120 ...
    python -m tools.acoustic_calibration --condition voz_tv --duration 120 ...
    python -m tools.acoustic_calibration --analyze data/calibration
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from tools.calibration_authority import CalibrationAuthorityClient

CHUNK_SIZE = 1280  # 80ms @ 16kHz — mismo framing que prod (multi_room_audio_loop)
SAMPLE_RATE = 16000
SPENERGY_POLL_S = 0.04  # ~25Hz, igual que el poller del gate
CONDITIONS = ("silencio", "tv", "voz", "voz_tv", "secador", "voz_secador")
SIGNALS = ("rms", "wake", "spenergy")


# ---------------------------------------------------------------- análisis

def summarize(samples: list[float]) -> dict:
    """Percentiles p5/p50/p95 + max de una lista de muestras."""
    if not samples:
        return {"count": 0, "p5": None, "p50": None, "p95": None, "max": None}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "count": int(arr.size),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def signal_gap(voice: list[float], ambient: list[float]) -> dict:
    """Gap = voz(p5) - ambiente(p95). gap > 0 → la señal separa voz de ambiente.

    El umbral recomendado es el punto medio del gap: máximo margen simétrico
    contra ambos lados de la distribución medida.
    """
    v, a = summarize(voice), summarize(ambient)
    if not v["count"] or not a["count"]:
        return {
            "separable": False, "gap": None, "recommended_threshold": None,
            "voice": v, "ambient": a,
        }
    gap = v["p5"] - a["p95"]
    separable = gap > 0
    threshold = (v["p5"] + a["p95"]) / 2.0 if separable else None
    return {
        "separable": separable, "gap": gap,
        "recommended_threshold": threshold, "voice": v, "ambient": a,
    }


def load_condition(directory: Path, condition: str) -> dict[str, list[float]]:
    """Carga todas las muestras de una condición desde los JSONL del directorio.

    Matchea archivos cuyo stem termine en ``_<condition>`` exactamente.
    Ignora filas meta (sin "kind"). El match exacto evita que *_tv.jsonl
    absorba archivos *_voz_tv.jsonl y viceversa.
    """
    out: dict[str, list[float]] = {s: [] for s in SIGNALS}
    # Usamos glob amplio *.jsonl y filtramos el stem exactamente.
    # El nombre de archivo tiene formato "<prefijo>_<condicion>.jsonl".
    # La condición puede contener '_' (p.ej. "voz_tv"), por lo que NO podemos
    # usar rsplit ni tokens finales para discriminar "tv" de "voz_tv" — ambos
    # terminan en el token "tv". La solución correcta: para cada archivo,
    # extraer el stem y buscar si alguna condición conocida hace que el stem sea
    # exactamente "<prefijo>_<condicion>" con prefijo no vacío. Comparamos el
    # stem con f"*_{condition}" anclado: el stem debe terminar en f"_{condition}"
    # y el carácter inmediatamente anterior al '_' separador NO debe ser parte
    # de otra condición más larga que también termine en condition.
    # Implementación concreta: el stem termina en f"_{condition}" y la longitud
    # del resto (prefijo) es > 0 Y el prefijo no termina a su vez en
    # f"_{condition}" anidado — en la práctica simplemente verificamos que el
    # stem[:-len(f'_{condition}')] no coincida con el nombre de otra condición
    # más larga. La manera más sencilla y robusta para nuestro dominio acotado
    # de condiciones: comparar el stem contra f"*_{cond}" para TODAS las
    # condiciones más largas que incluyen a `condition` como sufijo, y rechazar.
    anchor = f"_{condition}"
    # Condiciones que tienen a `condition` como sufijo propio (más largas).
    longer_conds = [c for c in CONDITIONS if c != condition and c.endswith(condition)]
    for f in sorted(Path(directory).glob("*.jsonl")):
        stem = f.stem
        # El stem debe terminar exactamente en "_<condition>".
        if not stem.endswith(anchor):
            continue
        # Rechazar si el stem también termina en "_<longer_condition>".
        if any(stem.endswith(f"_{lc}") for lc in longer_conds):
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("kind")
            if kind in out:
                out[kind].append(float(row["value"]))
    return out


# ---------------------------------------------------------------- captura

_CAPTURES: set = set()

_FAILED_CAPTURES: set = set()

_FAILED_CAPTURES_LOCK = threading.Lock()

class _CaptureOwner:
    """Own capture admission and actual retirement before publishing JSONL."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.rows: list[dict] = []
        self.detector = self.controller = self.stream = self.poller = None
        self.poller_handle = None
        self.poller_exited = threading.Event()
        self.poller_start_entered = False
        self.callbacks = 0
        self.sealed = False
        self.retired = False
        self.uncertain = False
        self.failures: list[BaseException] = []
        self.controls: list[BaseException] = []
        self.retirement_attempted = False
        # Retain before any SDK/model operation, including an interruption in
        # cleanup metadata. GC or a retained exception is never the owner.
        with _FAILED_CAPTURES_LOCK:
            _CAPTURES.add(self)

    def publish(self, row: dict) -> None:
        with self.condition:
            if not self.sealed:
                self.rows.append(row)

    def admit_callback(self) -> bool:
        with self.condition:
            if self.sealed:
                return False
            self.callbacks += 1
            return True

    def finish_callback(self) -> None:
        with self.condition:
            self.callbacks -= 1
            self.condition.notify_all()

    def start_poller(self) -> None:
        self.poller = threading.Thread(target=self._poll, daemon=True)
        handle = getattr(self.poller, "_handle", None)
        handle_type = getattr(threading, "_ThreadHandle", None)
        started = getattr(self.poller, "_started", None)
        if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13)
                or handle_type is None or type(handle) is not handle_type
                or type(getattr(handle, "ident", None)) is not int or handle.ident != 0
                or not callable(getattr(handle, "join", None))
                or not callable(getattr(handle, "is_done", None))
                or not callable(getattr(started, "is_set", None)) or started.is_set()):
            raise RuntimeError("calibration_poller_unsupported_runtime")
        self.poller_handle = handle
        self.poller_start_entered = True
        self.poller.start()

    def record_worker_failure(self, error: BaseException, code: str) -> None:
        # Keep no SDK traceback, context or payload in the process owner.
        if isinstance(error, Exception):
            failure = RuntimeError(code)
        else:
            try:
                if isinstance(error, SystemExit):
                    exit_code = error.code
                    if exit_code is not None and type(exit_code) is not int:
                        exit_code = 1
                    failure = type(error)(exit_code)
                else:
                    failure = type(error)()
            except BaseException:
                failure = BaseException(code)
        with self.condition:
            self.failures.append(failure)
            if not isinstance(failure, Exception):
                self.controls.append(failure)
            self.sealed = True
            self.stop.set()

    def _poll(self) -> None:
        try:
            _spenergy_poller(self.controller, self.rows, self.stop, publish=self.publish)
        except BaseException as error:
            self.record_worker_failure(error, "calibration_poller_failed")
        finally:
            self.poller_exited.set()

    def _attempt(self, function, *, uncertain: bool = False) -> bool:
        try:
            function()
            return True
        except BaseException as error:
            self.failures.append(error)
            if not isinstance(error, Exception):
                self.controls.append(error)
            if uncertain:
                self.uncertain = True
            return False

    def _join_poller(self) -> bool:
        if self.poller is None or not self.poller_start_entered:
            return True
        handle = self.poller_handle
        if handle is None or handle.ident == 0:
            # An exception from start does not prove no native dispatch. Keep
            # this owner reachable rather than deriving exit from ident=None.
            self.uncertain = True
            self.failures.append(RuntimeError("calibration_poller_start_unconfirmed"))
            return False
        while True:
            try:
                if self.poller is threading.current_thread():
                    raise RuntimeError("calibration_poller_self_join")
                # Same CPython 3.13 native join boundary as _PinnedWorker in
                # stt.pinned_executor, without coupling calibration to STT.
                if not self.poller._started.is_set():
                    handle.join()
                else:
                    self.poller.join()
                if not handle.is_done() or not self.poller_exited.is_set():
                    raise RuntimeError("calibration_poller_exit_unconfirmed")
                self.poller = self.poller_handle = None
                return True
            except BaseException as error:
                if not isinstance(error, Exception):
                    self.controls.append(error)
                    continue
                self.failures.append(error)
                self.uncertain = True
                return False

    def _wait_callbacks(self) -> bool:
        while True:
            try:
                with self.condition:
                    if not self.callbacks:
                        return True
                    self.condition.wait()
            except BaseException as error:
                if not isinstance(error, Exception):
                    self.controls.append(error)
                    continue
                self.failures.append(error)
                self.uncertain = True
                return False

    def retire(self, prior_error: BaseException | None = None) -> None:
        if self.retirement_attempted:
            raise RuntimeError("calibration_retirement_already_attempted")
        self.retirement_attempted = True
        with self.condition:
            self.sealed = True
            self.stop.set()
        if self.stream is not None:
            # InputStream.__exit__ uses ignore_errors=True and cannot provide
            # this receipt. Invoke the equivalent public operations directly,
            # without a duplicate __exit__/close or retry after close failure.
            active = None
            try:
                active = self.stream.active
            except AttributeError:
                pass
            except BaseException as error:
                self.failures.append(error)
                if not isinstance(error, Exception):
                    self.controls.append(error)
            if active is not False:
                self._attempt(lambda: self.stream.stop(ignore_errors=False))
            if self._attempt(lambda: self.stream.close(ignore_errors=False), uncertain=True):
                self.stream = None
        poller_done = self._join_poller()
        callbacks_done = self._wait_callbacks()
        if poller_done and callbacks_done and self.detector is not None:
            if self._attempt(self.detector.close, uncertain=True):
                self.detector = None
        if poller_done and self.controller is not None:
            # This drops Python references/cache only; it is not proof of a
            # physical PyUSB/libusb handle being released.
            if self._attempt(self.controller.invalidate_device, uncertain=True):
                self.controller = None
        self.retired = not self.uncertain
        if self.uncertain:
            # A traceback or GC is not a durable owner of a failed resource.
            with _FAILED_CAPTURES_LOCK:
                _FAILED_CAPTURES.add(self)
        else:
            with _FAILED_CAPTURES_LOCK:
                _CAPTURES.discard(self)
        if prior_error is not None and not isinstance(prior_error, Exception):
            raise prior_error
        if self.controls:
            raise self.controls[0]
        if self.failures:
            errors = ([prior_error] if prior_error is not None else []) + self.failures
            if len(errors) == 1:
                raise errors[0]
            raise ExceptionGroup("calibration_capture_failed", errors)

    def snapshot(self) -> list[dict]:
        with self.condition:
            if not self.retired or self.failures:
                raise RuntimeError("calibration_retirement_unconfirmed")
            return list(self.rows)

def _spenergy_poller(controller, rows: list, stop: threading.Event, *, publish=None) -> None:
    """Thread: pollea SPENERGY[3] cada 40ms y acumula filas. Fail-open."""
    while not stop.is_set():
        vals = row = None
        try:
            vals = controller.read_spenergy()
            if vals is not None:
                row = {"t": time.time(), "kind": "spenergy", "value": float(vals[3])}
                if publish is not None:
                    publish(row)
                elif not stop.is_set():
                    rows.append(row)
        except Exception as exc:
            print(f"⚠️  SPENERGY poller error: {exc} — datos SPENERGY incompletos")
            break
        finally:
            vals = row = None
        stop.wait(SPENERGY_POLL_S)


def _finish_authority(authority: CalibrationAuthorityClient, owner: _CaptureOwner,
                      prior_error: BaseException | None) -> None:
    """Observe local control-plane exit before issuing a retire-only receipt."""
    failures: list[BaseException] = []
    monitor_done = False
    try:
        authority.stop_monitor()
        monitor_done = True
    except BaseException as error:
        failures.append(error)
    if owner.retired and monitor_done:
        try:
            authority.retire()
        except BaseException as error:
            failures.append(error)
    try:
        authority.close()
    except BaseException as error:
        failures.append(error)
    if failures:
        errors = ([prior_error] if prior_error is not None else []) + failures
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup("calibration_authority_cleanup_failed", errors)

def run_capture(
    condition: str,
    duration_s: float,
    device: int,
    channel: int,
    model_path: str,
    out_dir: Path,
    usb_port: str | None = None,
    *,
    authority: CalibrationAuthorityClient | None = None,
) -> Path:
    """Capture scalar measurements, publishing JSONL only after real retirement.

    InputStream construction has no public pre-return resource receipt. A
    constructor failure still seals callbacks and retires all previously
    acquired resources; this tool cannot claim disposal of an unreturned SDK
    object or physical libusb internals.
    """
    if not isinstance(authority, CalibrationAuthorityClient):
        raise RuntimeError("calibration_authority_required")
    owner = _CaptureOwner()

    def seal() -> None:
        with owner.condition:
            owner.sealed = True
            owner.stop.set()

    def require_current() -> None:
        if not authority.current():
            seal()
            raise RuntimeError("calibration_capture_stopped")

    try:
        try:
            authority.register()
            authority.activate()
            authority.start_monitor(seal)
            require_current()
            import sounddevice as sd
            from src.wakeword.detector import WakeWordDetector
            from src.audio.xvf_controller import XvfController
            # Own each object before calling operations that load/open native work.
            owner.detector = WakeWordDetector(models=[model_path])
            require_current()
            owner.detector.load()
            require_current()
            owner.controller = XvfController(usb_port=usb_port)
            require_current()
            spenergy_ok = owner.controller.open()
            require_current()
            if not spenergy_ok:
                print("⚠️  XVF3800 no accesible — se mide sin SPENERGY (RMS + wake igual sirven)")
            owner.publish({
                "meta": True, "condition": condition, "device": device,
                "channel": channel, "model": model_path, "duration_s": duration_s,
                "started_at": time.time(),
            })
            if spenergy_ok:
                owner.start_poller()

            def callback(indata, frames, time_info, status):
                if not owner.admit_callback():
                    return
                chunk = scores = None
                try:
                    if status:
                        print(f"⚠️ sd callback status: {status}", flush=True)
                    ch = channel if indata.shape[1] > channel else 0
                    chunk = indata[:, ch].copy()
                    now = time.time()
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    owner.publish({"t": now, "kind": "rms", "value": rms})
                    scores = owner.detector.predict(chunk)
                    if scores:
                        owner.publish({"t": now, "kind": "wake", "value": float(max(scores.values()))})
                except BaseException as error:
                    owner.record_worker_failure(error, "calibration_callback_failed")
                finally:
                    # Discard owned PCM/driver parameters before publishing count0.
                    # Stream.close's strict receipt additionally covers SDK callers.
                    del indata, chunk, scores, time_info, status
                    owner.finish_callback()

            if channel == 0:
                print(
                    "⚠️  --channel 0: el XVF3800 tiene 2 canales — abrirlo como 1ch puede "
                    "garbler el audio. Considera usar --channel 1 o superior en dispositivos "
                    "multi-canal (ver _resolve_capture_channels en src/pipeline/multi_room_audio_loop.py)."
                )
            print(f"▶ Capturando condición '{condition}' por {duration_s:.0f}s "
                  f"(device={device}, channel={channel})...")
            require_current()
            try:
                owner.stream = sd.InputStream(
                    device=device, samplerate=SAMPLE_RATE,
                    channels=max(channel + 1, 1), dtype="float32", blocksize=CHUNK_SIZE,
                    callback=callback,
                )
            except BaseException:
                # No returned SDK object means no strict close receipt. Preserve
                # uncertainty rather than retiring the remote producer capability.
                owner.uncertain = True
                raise
            require_current()
            owner.stream.__enter__()
            require_current()
            deadline = time.time() + duration_s
            while time.time() < deadline and not owner.stop.is_set():
                time.sleep(1.0)
                elapsed = duration_s - (deadline - time.time())
                print(f"  ... {elapsed:.0f}/{duration_s:.0f}s ({len(owner.rows)} muestras)", end="\r")
        finally:
            owner.retire(sys.exc_info()[1])
        require_current()
        snapshot = owner.snapshot()

        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_file = out_dir / f"{stamp}_{condition}.jsonl"
        require_current()
        out_file.write_text("\n".join(json.dumps(r) for r in snapshot))
        require_current()
        print(f"\n✔ {len(snapshot)} muestras → {out_file}")
        return out_file
    finally:
        _finish_authority(authority, owner, sys.exc_info()[1])


# ---------------------------------------------------------------- reporte

def print_report(directory: Path) -> None:
    """Tabla por señal × condición + veredicto de separabilidad voz-vs-tv."""
    data = {c: load_condition(directory, c) for c in CONDITIONS}

    for signal in SIGNALS:
        print(f"\n=== {signal.upper()} ===")
        print(f"{'condición':<10} {'n':>6} {'p5':>12} {'p50':>12} {'p95':>12} {'max':>12}")
        for cond in CONDITIONS:
            s = summarize(data[cond][signal])
            if s["count"] == 0:
                print(f"{cond:<10} {'—':>6}")
                continue
            print(f"{cond:<10} {s['count']:>6} {s['p5']:>12.4f} {s['p50']:>12.4f} "
                  f"{s['p95']:>12.4f} {s['max']:>12.4f}")
        # Veredicto: ¿separa la voz del ambiente? Por cada fuente de ruido
        # medida (tv, secador) comparamos voz limpia y voz embebida en él.
        pairs = (
            ("voz", "tv"), ("voz_tv", "tv"),
            ("voz", "secador"), ("voz_secador", "secador"),
        )
        for voice_cond, ambient_cond in pairs:
            g = signal_gap(data[voice_cond][signal], data[ambient_cond][signal])
            verdict = (
                f"SEPARABLE — umbral recomendado {g['recommended_threshold']:.4f}"
                if g["separable"] else "NO separa"
            )
            if g["gap"] is not None:
                print(f"  {voice_cond} vs {ambient_cond}: gap={g['gap']:+.4f} → {verdict}")
            else:
                print(f"  {voice_cond} vs {ambient_cond}: sin datos")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--device", type=int, help="índice sounddevice del mic (prod usa binding mic_usb_port; ver logs de kza-voice o sd.query_devices())")
    parser.add_argument("--channel", type=int, default=1, help="canal de captura (prod escritorio=1 ASR)")
    # Default = path absoluto de prod (settings.yaml rooms.wake_word.model);
    # un path relativo solo resuelve desde el app dir y falla en detector.load().
    parser.add_argument("--model", default="/home/kza/kza/models/wakeword/nexa.onnx")
    parser.add_argument("--out", default="data/calibration")
    parser.add_argument(
        "--port",
        metavar="BUS-PUERTOS",
        help=(
            'puerto USB del mic ("5-5.4"), mismo formato que '
            "rooms.<room>.mic_usb_port. SIN esto se polea el SPENERGY del "
            "primer XVF3800 que enumere — con más de un mic conectado eso "
            "calibra una habitación con la energía acústica de otra"
        ),
    )
    parser.add_argument("--dashboard-url", help="origen explícito del Dashboard que controla esta captura")
    parser.add_argument("--token-env", default="KZA_DASHBOARD_TOKEN",
                        help="variable de entorno de la credencial administrativa; no pasar el token como argumento")
    parser.add_argument("--analyze", metavar="DIR", help="solo análisis de JSONLs existentes")
    args = parser.parse_args(argv)

    if args.analyze:
        print_report(Path(args.analyze))
        return 0
    if not args.condition or args.device is None:
        parser.error("--condition y --device son requeridos para capturar (o usar --analyze)")
    if not args.dashboard_url:
        parser.error("--dashboard-url es requerido para capturar")
    token = os.environ.get(args.token_env)
    if not token:
        parser.error("No hay credencial administrativa en la variable indicada")
    authority = CalibrationAuthorityClient(args.dashboard_url, token)
    token = None
    run_capture(args.condition, args.duration, args.device, args.channel,
                args.model, Path(args.out), usb_port=args.port, authority=authority)
    return 0


if __name__ == "__main__":
    sys.exit(main())
