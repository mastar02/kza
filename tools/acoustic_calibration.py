"""Harness de calibración acústica — matriz voz/TV/silencio (spec 2026-06-05).

Mide simultáneamente, bajo una condición etiquetada (silencio/tv/voz/voz_tv):
  - RMS por chunk del stream del mic (mismo device/canal que prod)
  - score máximo de openwakeword (nexa.onnx) por chunk
  - SPENERGY[3] del XVF3800 (poll 25Hz vía XvfController, fail-open)

La pregunta que responde: ¿qué señal tiene gap entre voz(p5) y tv(p95)?
Esa señal es compuerta viable; las que no separan quedan documentadas como
muertas con el chip en su estado actual (MAXGAIN=8, ch1 ASR).

⚠️ Correr con kza-voice PARADO (contención del mic y del USB vendor):
    systemctl --user stop kza-voice

Uso (server, venv de kza):
    python -m tools.acoustic_calibration --condition silencio --duration 120 \
        --device 4 --channel 1 --model models/wakeword/nexa.onnx
    python -m tools.acoustic_calibration --condition tv --duration 180 ...
    python -m tools.acoustic_calibration --condition voz --duration 120 ...
    python -m tools.acoustic_calibration --condition voz_tv --duration 120 ...
    python -m tools.acoustic_calibration --analyze data/calibration
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

CHUNK_SIZE = 1280  # 80ms @ 16kHz — mismo framing que prod (multi_room_audio_loop)
SAMPLE_RATE = 16000
SPENERGY_POLL_S = 0.04  # ~25Hz, igual que el poller del gate
CONDITIONS = ("silencio", "tv", "voz", "voz_tv")
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

def _spenergy_poller(controller, rows: list, stop: threading.Event) -> None:
    """Thread: pollea SPENERGY[3] cada 40ms y acumula filas. Fail-open."""
    while not stop.is_set():
        vals = controller.read_spenergy()
        if vals is not None:
            rows.append({"t": time.time(), "kind": "spenergy", "value": float(vals[3])})
        time.sleep(SPENERGY_POLL_S)


def run_capture(
    condition: str,
    duration_s: float,
    device: int,
    channel: int,
    model_path: str,
    out_dir: Path,
) -> Path:
    """Captura RMS + score wake + SPENERGY durante duration_s. Devuelve el JSONL."""
    import sounddevice as sd

    from src.wakeword.detector import WakeWordDetector
    from src.audio.xvf_controller import XvfController

    detector = WakeWordDetector(models=[model_path], threshold=1.1)  # >1 → nunca "detecta", solo medimos scores
    detector.load()

    controller = XvfController()
    spenergy_ok = controller.open()
    if not spenergy_ok:
        print("⚠️  XVF3800 no accesible — se mide sin SPENERGY (RMS + wake igual sirven)")

    rows: list[dict] = []
    rows.append({
        "meta": True, "condition": condition, "device": device,
        "channel": channel, "model": model_path, "duration_s": duration_s,
        "started_at": time.time(),
    })

    stop = threading.Event()
    poller = None
    if spenergy_ok:
        poller = threading.Thread(
            target=_spenergy_poller, args=(controller, rows, stop), daemon=True
        )
        poller.start()

    def callback(indata, frames, time_info, status):
        ch = channel if indata.shape[1] > channel else 0
        chunk = indata[:, ch].copy()
        now = time.time()
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        rows.append({"t": now, "kind": "rms", "value": rms})
        scores = detector.predict(chunk)
        if scores:
            rows.append({"t": now, "kind": "wake", "value": float(max(scores.values()))})

    print(f"▶ Capturando condición '{condition}' por {duration_s:.0f}s "
          f"(device={device}, channel={channel})...")
    stream = sd.InputStream(
        device=device, samplerate=SAMPLE_RATE, channels=channel + 1 if channel else 1,
        dtype="float32", blocksize=CHUNK_SIZE, callback=callback,
    )
    try:
        with stream:
            deadline = time.time() + duration_s
            while time.time() < deadline:
                time.sleep(1.0)
                elapsed = duration_s - (deadline - time.time())
                print(f"  ... {elapsed:.0f}/{duration_s:.0f}s ({len(rows)} muestras)", end="\r")
    finally:
        stop.set()
        if poller is not None:
            poller.join(timeout=1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_file = out_dir / f"{stamp}_{condition}.jsonl"
    out_file.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"\n✔ {len(rows)} muestras → {out_file}")
    return out_file


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
        # Veredicto: ¿separa la voz (con y sin TV) del ambiente TV?
        for voice_cond in ("voz", "voz_tv"):
            g = signal_gap(data[voice_cond][signal], data["tv"][signal])
            verdict = (
                f"SEPARABLE — umbral recomendado {g['recommended_threshold']:.4f}"
                if g["separable"] else "NO separa"
            )
            if g["gap"] is not None:
                print(f"  {voice_cond} vs tv: gap={g['gap']:+.4f} → {verdict}")
            else:
                print(f"  {voice_cond} vs tv: sin datos")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--device", type=int, help="índice sounddevice del mic (prod usa binding mic_usb_port; ver logs de kza-voice o sd.query_devices())")
    parser.add_argument("--channel", type=int, default=1, help="canal de captura (prod escritorio=1 ASR)")
    parser.add_argument("--model", default="models/wakeword/nexa.onnx")
    parser.add_argument("--out", default="data/calibration")
    parser.add_argument("--analyze", metavar="DIR", help="solo análisis de JSONLs existentes")
    args = parser.parse_args(argv)

    if args.analyze:
        print_report(Path(args.analyze))
        return 0
    if not args.condition or args.device is None:
        parser.error("--condition y --device son requeridos para capturar (o usar --analyze)")
    run_capture(args.condition, args.duration, args.device, args.channel,
                args.model, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
