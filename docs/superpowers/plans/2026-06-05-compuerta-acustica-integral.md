# Compuerta Acústica Integral — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restaurar el control acústico del pipeline con TV de fondo: harness de calibración medición-primero + `AmbientGuard` (escalera NORMAL/STRICT/COOLDOWN por habitación) que garantiza cero saturación del router y endurece las compuertas bajo ambiente hostil.

**Architecture:** Un harness standalone (`tools/acoustic_calibration.py`) mide RMS/score-wake/SPENERGY bajo 4 condiciones (silencio/tv/voz/voz_tv) y recomienda umbrales con datos reales del chip (MAXGAIN=8, ch1 ASR). Un componente puro nuevo (`src/pipeline/ambient_guard.py`) implementa la máquina de estados por room, alimentada por la tasa de capturas rechazadas (señal de software, independiente del chip) y enforced en `_should_accept_wakeword()`. En STRICT se desactiva además el bonus `wake_acoustically_confirmed` del grammar fast-path (vía flag `ambient_strict` en `CommandEvent`).

**Tech Stack:** Python 3.13, pytest, numpy, sounddevice, openwakeword, pyusb (XvfController existente). Spec: `docs/superpowers/specs/2026-06-05-compuerta-acustica-integral-design.md`.

**Convenciones del repo (LEER ANTES DE EMPEZAR):**
- venv local: `/Users/yo/Documents/kza/.venv/bin/python` (el python3 del sistema es 3.9 y rompe con `slots=True`). Correr pytest como `.venv/bin/python -m pytest`.
- Los tests de `tests/unit/pipeline/` mockean `sys.modules` (torch/sounddevice) — correr los archivos de este plan AISLADOS, no en la suite completa (interferencia conocida).
- Commits frecuentes en `feat/nexa-command-detection-fixes`. Mensajes estilo `feat(pipeline): ...` / `fix(...)`. NO pushear sin pedido del usuario.
- Código/logs en inglés o español según el archivo circundante (este repo mezcla; los módulos nuevos siguen el estilo de `xvf_controller.py`: docstrings en español, identifiers en inglés).

---

## File Structure

| Archivo | Acción | Responsabilidad |
|---|---|---|
| `tools/acoustic_calibration.py` | Crear | Harness de captura + funciones de análisis (percentiles, gap, recomendación de umbral) + modo `--analyze` |
| `tests/unit/tools/test_acoustic_calibration.py` | Crear | Tests de las funciones de análisis |
| `src/pipeline/ambient_guard.py` | Crear | `GuardState`, `AmbientGuardConfig`, `GuardDecision`, `AmbientGuard`, `classify_outcome` |
| `tests/unit/pipeline/test_ambient_guard.py` | Crear | Tests de la máquina de estados + `classify_outcome` |
| `src/pipeline/command_event.py` | Modificar | Campo nuevo `ambient_strict: bool = False` |
| `src/pipeline/multi_room_audio_loop.py` | Modificar | Inyección del guard; enforcement en wake/follow_up/dispatch |
| `tests/unit/pipeline/test_multi_room_audio_loop.py` | Modificar | Tests de integración del guard en el loop |
| `src/pipeline/request_router.py` | Modificar | `ambient_strict` desactiva el bonus wake del grammar fast-path |
| `tests/unit/pipeline/test_request_router_grammar.py` | Modificar | Test del bonus desactivado en STRICT |
| `src/main.py` | Modificar | Construcción de `AmbientGuard` desde config + inyección al loop |
| `config/settings.yaml` | Modificar | Bloque `rooms.ambient_guard` (pasivo por default) y luego umbrales medidos |

---

### Task 1: Funciones de análisis de calibración

**Files:**
- Create: `tools/acoustic_calibration.py`
- Test: `tests/unit/tools/test_acoustic_calibration.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests de las funciones de análisis del harness de calibración acústica.

Solo cubren la parte PURA (percentiles, gap voz-vs-ambiente, recomendación
de umbral, carga de JSONL). El loop de captura (sounddevice/openwakeword/
XvfController) es glue fino validado a mano en el server.
"""
import json

import pytest

from tools.acoustic_calibration import (
    summarize,
    signal_gap,
    load_condition,
)


class TestSummarize:
    def test_empty_samples(self):
        s = summarize([])
        assert s == {"count": 0, "p5": None, "p50": None, "p95": None, "max": None}

    def test_known_distribution(self):
        # 0..99 → p50 ≈ 49.5, max = 99
        s = summarize(list(range(100)))
        assert s["count"] == 100
        assert s["max"] == 99.0
        assert 48.0 <= s["p50"] <= 51.0
        assert 3.0 <= s["p5"] <= 6.0
        assert 93.0 <= s["p95"] <= 96.0


class TestSignalGap:
    def test_separable_signal(self):
        # Voz claramente arriba del ambiente: voz p5 > ambiente p95
        voice = [100.0 + i for i in range(50)]      # 100..149
        ambient = [1.0 + i * 0.1 for i in range(50)]  # 1..5.9
        g = signal_gap(voice, ambient)
        assert g["separable"] is True
        assert g["gap"] > 0
        # Umbral recomendado en el medio del gap
        assert g["ambient"]["p95"] < g["recommended_threshold"] < g["voice"]["p5"]

    def test_overlapping_signal_not_separable(self):
        voice = [10.0 + i * 0.1 for i in range(50)]
        ambient = [9.0 + i * 0.1 for i in range(50)]  # solapa con voz
        g = signal_gap(voice, ambient)
        assert g["separable"] is False
        assert g["recommended_threshold"] is None

    def test_empty_side_not_separable(self):
        g = signal_gap([], [1.0, 2.0])
        assert g["separable"] is False
        assert g["gap"] is None


class TestLoadCondition:
    def test_groups_rows_by_kind(self, tmp_path):
        rows = [
            {"t": 1.0, "kind": "rms", "value": 0.01},
            {"t": 1.0, "kind": "wake", "value": 0.3},
            {"t": 1.1, "kind": "spenergy", "value": 0.0},
            {"t": 1.2, "kind": "rms", "value": 0.02},
        ]
        f = tmp_path / "20260605_tv.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in rows))
        data = load_condition(tmp_path, "tv")
        assert data["rms"] == [0.01, 0.02]
        assert data["wake"] == [0.3]
        assert data["spenergy"] == [0.0]

    def test_missing_condition_returns_empty(self, tmp_path):
        data = load_condition(tmp_path, "voz")
        assert data == {"rms": [], "wake": [], "spenergy": []}

    def test_ignores_meta_rows(self, tmp_path):
        rows = [
            {"meta": True, "condition": "tv", "device": 4},
            {"t": 1.0, "kind": "rms", "value": 0.5},
        ]
        f = tmp_path / "20260605_tv.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in rows))
        data = load_condition(tmp_path, "tv")
        assert data["rms"] == [0.5]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/tools/test_acoustic_calibration.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'tools.acoustic_calibration'`

- [ ] **Step 3: Write the analysis functions**

Crear `tools/acoustic_calibration.py` (por ahora solo la parte pura; el CLI llega en Task 2):

```python
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

    Matchea archivos ``*_<condition>.jsonl``. Ignora filas meta (sin "kind").
    """
    out: dict[str, list[float]] = {s: [] for s in SIGNALS}
    for f in sorted(Path(directory).glob(f"*_{condition}.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("kind")
            if kind in out:
                out[kind].append(float(row["value"]))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/tools/test_acoustic_calibration.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/acoustic_calibration.py tests/unit/tools/test_acoustic_calibration.py
git commit -m "feat(tools): funciones de análisis de calibración acústica (percentiles + gap voz/ambiente)"
```

---

### Task 2: Harness de captura + modo --analyze (CLI)

**Files:**
- Modify: `tools/acoustic_calibration.py` (agregar captura y CLI al final del archivo)

Glue fino sin tests unitarios (sounddevice/openwakeword/USB reales); se valida a mano en el server en Task 8. Las funciones de análisis ya están cubiertas por Task 1.

- [ ] **Step 1: Agregar captura, reporte y main() al archivo**

Append a `tools/acoustic_calibration.py`:

```python
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
```

- [ ] **Step 2: Verificar que los tests de Task 1 siguen verdes y el CLI parsea**

Run: `.venv/bin/python -m pytest tests/unit/tools/test_acoustic_calibration.py -v && .venv/bin/python -m tools.acoustic_calibration --help`
Expected: tests PASS + help del CLI sin traceback (sounddevice se importa lazy dentro de run_capture, así el --help funciona en la laptop sin mic)

- [ ] **Step 3: Commit**

```bash
git add tools/acoustic_calibration.py
git commit -m "feat(tools): harness de captura de calibración acústica (RMS+wake+SPENERGY, 4 condiciones)"
```

---

### Task 3: AmbientGuard — máquina de estados (núcleo)

**Files:**
- Create: `src/pipeline/ambient_guard.py`
- Test: `tests/unit/pipeline/test_ambient_guard.py`

Semántica exacta (decisiones del diseño, NO improvisar):
- Escalada NORMAL→STRICT y STRICT→COOLDOWN la disparan SOLO los resultados de captura rechazados (`noise`/`empty`/`timeout`) — cosas que ya gastaron Whisper/router. Los rechazos del propio guard en STRICT (score bajo) son gratis: NO cuentan para COOLDOWN, pero SÍ refrescan `last_reject_at` (el ambiente persiste → STRICT sigue vivo).
- `accepted` y `other_fail` (voz real con fallo downstream, p.ej. error HA) NO escalan.
- Salida de STRICT→NORMAL: `strict_exit_quiet_s` sin ningún rechazo (histéresis).
- COOLDOWN expira a STRICT (no a NORMAL): tras la tormenta se sigue estricto hasta el quiet.
- Transiciones por tiempo se evalúan lazy en cada llamada (no hay timers).
- `on_wake` corre en el thread C de sounddevice y `on_capture_result` en el event loop → `threading.Lock` interno.
- `enabled=False` (default) → guard 100% pasivo: acepta todo, no acumula estado.

- [ ] **Step 1: Write the failing tests**

Crear `tests/unit/pipeline/test_ambient_guard.py`:

```python
"""Tests del AmbientGuard — compuerta acústica integral (spec 2026-06-05).

Máquina de estados NORMAL → STRICT → COOLDOWN por habitación, alimentada por
la tasa de capturas rechazadas. Reloj inyectado: cero sleeps.
"""
import pytest

from src.pipeline.ambient_guard import (
    AmbientGuard,
    AmbientGuardConfig,
    GuardState,
    classify_outcome,
)


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_guard(clock=None, **overrides) -> AmbientGuard:
    cfg = AmbientGuardConfig(
        enabled=True,
        strict_entry_rejects=3,
        strict_entry_window_s=60.0,
        strict_exit_quiet_s=120.0,
        strict_wake_score=0.65,
        strict_min_rms=0.0,
        strict_min_spenergy=0.0,
        cooldown_entry_rejects=3,
        cooldown_entry_window_s=60.0,
        cooldown_duration_s=30.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return AmbientGuard(config=cfg, time_fn=clock or FakeClock())


class TestPassiveDefaults:
    def test_disabled_accepts_everything(self):
        guard = AmbientGuard()  # config default: enabled=False
        d = guard.on_wake("escritorio", score=0.01, rms=0.0)
        assert d.accept is True
        assert d.reason == "disabled"

    def test_disabled_never_escalates(self):
        guard = AmbientGuard()
        for _ in range(50):
            guard.on_capture_result("escritorio", "noise")
        assert guard.state_for("escritorio") is GuardState.NORMAL

    def test_disabled_follow_up_allowed(self):
        guard = AmbientGuard()
        assert guard.follow_up_allowed("escritorio") is True


class TestNormalState:
    def test_accepts_any_score_in_normal(self):
        guard = make_guard()
        d = guard.on_wake("escritorio", score=0.41, rms=0.001)
        assert d.accept is True
        assert d.state is GuardState.NORMAL

    def test_follow_up_allowed_in_normal(self):
        guard = make_guard()
        assert guard.follow_up_allowed("escritorio") is True


class TestEscalationToStrict:
    def test_rejects_within_window_escalate(self):
        guard = make_guard()
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")
        assert guard.state_for("escritorio") is GuardState.STRICT

    def test_rejects_outside_window_do_not_escalate(self):
        clock = FakeClock()
        guard = make_guard(clock=clock)
        guard.on_capture_result("escritorio", "noise")
        clock.advance(61.0)
        guard.on_capture_result("escritorio", "noise")
        clock.advance(61.0)
        guard.on_capture_result("escritorio", "noise")
        assert guard.state_for("escritorio") is GuardState.NORMAL

    def test_accepted_and_other_fail_do_not_escalate(self):
        guard = make_guard()
        for outcome in ("accepted", "other_fail", "accepted", "other_fail",
                        "accepted", "other_fail"):
            guard.on_capture_result("escritorio", outcome)
        assert guard.state_for("escritorio") is GuardState.NORMAL

    def test_all_reject_kinds_count(self):
        guard = make_guard()
        for outcome in ("noise", "empty", "timeout"):
            guard.on_capture_result("escritorio", outcome)
        assert guard.state_for("escritorio") is GuardState.STRICT


class TestStrictState:
    def _strict_guard(self, clock=None, **overrides):
        guard = make_guard(clock=clock, **overrides)
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")
        assert guard.state_for("escritorio") is GuardState.STRICT
        return guard

    def test_low_score_rejected_in_strict(self):
        guard = self._strict_guard()
        d = guard.on_wake("escritorio", score=0.50, rms=0.05)
        assert d.accept is False
        assert d.reason == "strict_score"
        assert d.state is GuardState.STRICT

    def test_high_score_accepted_in_strict(self):
        guard = self._strict_guard()
        d = guard.on_wake("escritorio", score=0.80, rms=0.05)
        assert d.accept is True

    def test_strict_min_rms_enforced_when_configured(self):
        guard = self._strict_guard(strict_min_rms=0.02)
        d = guard.on_wake("escritorio", score=0.80, rms=0.01)
        assert d.accept is False
        assert d.reason == "strict_rms"

    def test_strict_min_spenergy_enforced_when_configured(self):
        guard = self._strict_guard(strict_min_spenergy=50.0)
        d = guard.on_wake("escritorio", score=0.80, rms=0.05, spenergy_peak=10.0)
        assert d.accept is False
        assert d.reason == "strict_spenergy"

    def test_spenergy_none_fails_open(self):
        # Sin lectura del chip (fail-open del controller) NO se bloquea voz.
        guard = self._strict_guard(strict_min_spenergy=50.0)
        d = guard.on_wake("escritorio", score=0.80, rms=0.05, spenergy_peak=None)
        assert d.accept is True

    def test_follow_up_blocked_in_strict(self):
        guard = self._strict_guard()
        assert guard.follow_up_allowed("escritorio") is False

    def test_exit_to_normal_after_quiet(self):
        clock = FakeClock()
        guard = self._strict_guard(clock=clock)
        clock.advance(121.0)  # > strict_exit_quiet_s sin rechazos
        assert guard.state_for("escritorio") is GuardState.NORMAL

    def test_guard_rejection_keeps_strict_alive(self):
        # Los rechazos del propio guard (TV sigue disparando wakes con score
        # bajo) refrescan el quiet timer → STRICT no expira mientras haya TV.
        clock = FakeClock()
        guard = self._strict_guard(clock=clock)
        clock.advance(100.0)
        guard.on_wake("escritorio", score=0.50, rms=0.05)  # rechazo del guard
        clock.advance(100.0)  # 200s desde la escalada, pero 100s desde el último rechazo
        assert guard.state_for("escritorio") is GuardState.STRICT

    def test_accepted_command_does_not_exit_strict(self):
        # Un comando real exitoso con TV de fondo NO saca de STRICT (la TV
        # sigue ahí); la salida es solo por quiet sostenido.
        guard = self._strict_guard()
        guard.on_capture_result("escritorio", "accepted")
        assert guard.state_for("escritorio") is GuardState.STRICT


class TestCooldown:
    def _cooldown_guard(self, clock):
        guard = make_guard(clock=clock)
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # → STRICT
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # → COOLDOWN
        assert guard.state_for("escritorio") is GuardState.COOLDOWN
        return guard

    def test_capture_rejects_in_strict_escalate_to_cooldown(self):
        self._cooldown_guard(FakeClock())

    def test_everything_rejected_during_cooldown(self):
        guard = self._cooldown_guard(FakeClock())
        d = guard.on_wake("escritorio", score=0.99, rms=0.5)
        assert d.accept is False
        assert d.reason == "cooldown"

    def test_cooldown_expires_to_strict(self):
        clock = FakeClock()
        guard = self._cooldown_guard(clock)
        clock.advance(31.0)  # > cooldown_duration_s
        assert guard.state_for("escritorio") is GuardState.STRICT

    def test_guard_rejections_do_not_escalate_to_cooldown(self):
        # Rechazos a nivel guard (strict_score) son gratis: no gastan
        # Whisper/router → NO cuentan para COOLDOWN.
        guard = make_guard()
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # → STRICT
        for _ in range(10):
            guard.on_wake("escritorio", score=0.50, rms=0.05)
        assert guard.state_for("escritorio") is GuardState.STRICT


class TestPerRoomIsolation:
    def test_rooms_have_independent_state(self):
        guard = make_guard()
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")
        assert guard.state_for("escritorio") is GuardState.STRICT
        assert guard.state_for("living") is GuardState.NORMAL
        d = guard.on_wake("living", score=0.41, rms=0.01)
        assert d.accept is True


class TestClassifyOutcome:
    def test_success_is_accepted(self):
        assert classify_outcome({"success": True, "text": "prende la luz"}) == "accepted"

    def test_empty_text(self):
        assert classify_outcome({"success": False, "text": ""}) == "empty"
        assert classify_outcome({"success": False, "text": None}) == "empty"
        assert classify_outcome({}) == "empty"

    def test_gate_and_llm_rejections_are_noise(self):
        assert classify_outcome(
            {"success": False, "text": "x", "intent": "gate_rejected"}) == "noise"
        assert classify_outcome(
            {"success": False, "text": "x", "intent": "llm_rejected:tv_phrase"}) == "noise"
        assert classify_outcome(
            {"success": False, "text": "x", "intent": "low_confidence:0.40"}) == "noise"

    def test_unavailable_and_timeout_are_timeout(self):
        # El LLMRouter produce rejection_reason="unavailable" en timeout/error
        # local → intent "llm_rejected:unavailable" debe clasificar timeout,
        # NO noise (por eso este check va ANTES del de llm_rejected).
        assert classify_outcome(
            {"success": False, "text": "x", "intent": "llm_rejected:unavailable"}) == "timeout"
        assert classify_outcome(
            {"success": False, "text": "x", "intent": "timeout"}) == "timeout"

    def test_real_command_downstream_failure_is_other_fail(self):
        # Voz real que falló en HA: NO debe escalar el guard.
        assert classify_outcome(
            {"success": False, "text": "prende la luz", "intent": "domotics"}) == "other_fail"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_ambient_guard.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.pipeline.ambient_guard'`

- [ ] **Step 3: Write the implementation**

Crear `src/pipeline/ambient_guard.py`:

```python
"""AmbientGuard — compuerta acústica integral por habitación (spec 2026-06-05).

Con TV de fondo, el wake openwakeword dispara constantemente (0.4-0.9) y el
ambiente satura Whisper + LLM router ("no me escucha" = comandos reales
compitiendo contra la cola; 1 acción fantasma el 2026-06-04). Este guard
unifica TV-mode + circuit breaker en una escalera de 3 estados POR ROOM:

    NORMAL ──(rechazos de captura ≥ N en ventana)──► STRICT
    STRICT ──(rechazos persisten)──► COOLDOWN ──(expira)──► STRICT
    STRICT ──(quiet sostenido, histéresis)──► NORMAL

- NORMAL: comportamiento actual (threshold base del detector).
- STRICT: exige score de wake ≥ strict_wake_score (encima del threshold base
  del detector, que NO se muta) + RMS/SPENERGY mínimos si están calibrados.
  El follow_up queda deshabilitado (la cascada del 06-04 era follow_up
  siempre abierto). El bonus wake_acoustically_confirmed del grammar
  fast-path también se apaga (ver request_router).
- COOLDOWN: descarta toda captura por cooldown_duration_s. Garantiza que la
  cola del router NUNCA se satura, sea cual sea el estado de las señales
  acústicas (la señal de escalada es de software: capturas rechazadas).

Semántica de escalada (decisión de diseño, ver spec):
- Solo los RESULTADOS DE CAPTURA rechazados (noise/empty/timeout — ya
  gastaron Whisper/router) escalan. Los rechazos del propio guard en STRICT
  son gratis: no cuentan para COOLDOWN, pero refrescan el quiet timer
  (ambiente persiste → STRICT sigue vivo).
- accepted / other_fail (voz real con fallo downstream) no escalan.

Thread-safety: on_wake corre en el thread C de sounddevice;
on_capture_result en el event loop → lock interno. Reloj inyectable
(time_fn) para tests sin sleeps. enabled=False (default) = guard pasivo.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Outcomes de captura que evidencian ambiente hostil (gastaron pipeline y
# fueron basura). "accepted" y "other_fail" no escalan.
REJECT_OUTCOMES = ("noise", "empty", "timeout")


class GuardState(Enum):
    NORMAL = "normal"
    STRICT = "strict"
    COOLDOWN = "cooldown"


@dataclass
class AmbientGuardConfig:
    """Config del guard (settings.yaml → rooms.ambient_guard).

    Los umbrales acústicos (strict_wake_score / strict_min_rms /
    strict_min_spenergy) se calibran con tools/acoustic_calibration.py —
    NO adivinar (lección de las sesiones 05-31..06-04).
    """
    enabled: bool = False
    # Escalada NORMAL → STRICT: N capturas rechazadas dentro de la ventana.
    strict_entry_rejects: int = 4
    strict_entry_window_s: float = 60.0
    # Salida STRICT → NORMAL: histéresis de quiet (sin NINGÚN rechazo).
    strict_exit_quiet_s: float = 120.0
    # Compuertas en STRICT. 0.0 = no aplicar (señal no calibrada/muerta).
    strict_wake_score: float = 0.65
    strict_min_rms: float = 0.0
    strict_min_spenergy: float = 0.0
    # Escalada STRICT → COOLDOWN: N capturas rechazadas MÁS en la ventana.
    cooldown_entry_rejects: int = 6
    cooldown_entry_window_s: float = 60.0
    cooldown_duration_s: float = 30.0


@dataclass
class GuardDecision:
    accept: bool
    reason: str  # "ok" | "disabled" | "strict_score" | "strict_rms" | "strict_spenergy" | "cooldown"
    state: GuardState


@dataclass
class _RoomState:
    state: GuardState = GuardState.NORMAL
    capture_rejects: deque = field(default_factory=deque)  # timestamps
    last_reject_at: float = 0.0
    cooldown_until: float = 0.0


def classify_outcome(result: dict) -> str:
    """Mapea el dict resultado del RequestRouter a un outcome del guard.

    Returns:
        "accepted" | "empty" | "noise" | "timeout" | "other_fail"
    """
    if result.get("success"):
        return "accepted"
    text = (result.get("text") or "").strip()
    if not text:
        return "empty"
    intent = str(result.get("intent") or "")
    # "unavailable" la produce el LLMRouter en timeout/error local → puede
    # venir como "llm_rejected:unavailable": chequear ANTES que llm_rejected.
    if "timeout" in intent or "unavailable" in intent:
        return "timeout"
    if (
        intent == "gate_rejected"
        or intent.startswith("llm_rejected")
        or intent.startswith("low_confidence")
    ):
        return "noise"
    return "other_fail"


class AmbientGuard:
    """Escalera NORMAL/STRICT/COOLDOWN por room. Ver docstring del módulo."""

    def __init__(
        self,
        config: AmbientGuardConfig | None = None,
        time_fn=time.time,
    ):
        self.config = config or AmbientGuardConfig()
        self._time = time_fn
        self._rooms: dict[str, _RoomState] = {}
        self._lock = threading.Lock()

    # ---- API pública ----

    def on_wake(
        self,
        room_id: str,
        score: float,
        rms: float,
        spenergy_peak: float | None = None,
    ) -> GuardDecision:
        """Decisión sobre un wake detectado (llamado desde el audio thread).

        spenergy_peak=None = sin lectura del chip → fail-open (nunca se
        bloquea voz por un fallo USB).
        """
        if not self.config.enabled:
            return GuardDecision(True, "disabled", GuardState.NORMAL)
        now = self._time()
        with self._lock:
            rs = self._room(room_id)
            self._refresh(rs, now)
            if rs.state is GuardState.COOLDOWN:
                return GuardDecision(False, "cooldown", rs.state)
            if rs.state is GuardState.STRICT:
                if score < self.config.strict_wake_score:
                    rs.last_reject_at = now  # ambiente persiste → quiet timer se refresca
                    return GuardDecision(False, "strict_score", rs.state)
                if self.config.strict_min_rms > 0.0 and rms < self.config.strict_min_rms:
                    rs.last_reject_at = now
                    return GuardDecision(False, "strict_rms", rs.state)
                if (
                    self.config.strict_min_spenergy > 0.0
                    and spenergy_peak is not None
                    and spenergy_peak < self.config.strict_min_spenergy
                ):
                    rs.last_reject_at = now
                    return GuardDecision(False, "strict_spenergy", rs.state)
            return GuardDecision(True, "ok", rs.state)

    def on_capture_result(self, room_id: str, outcome: str) -> None:
        """Reporta el resultado de una captura ya procesada (event loop)."""
        if not self.config.enabled:
            return
        now = self._time()
        with self._lock:
            rs = self._room(room_id)
            self._refresh(rs, now)
            if outcome not in REJECT_OUTCOMES:
                return
            rs.last_reject_at = now
            rs.capture_rejects.append(now)
            window = (
                self.config.cooldown_entry_window_s
                if rs.state is GuardState.STRICT
                else self.config.strict_entry_window_s
            )
            cutoff = now - window
            while rs.capture_rejects and rs.capture_rejects[0] < cutoff:
                rs.capture_rejects.popleft()
            if (
                rs.state is GuardState.NORMAL
                and len(rs.capture_rejects) >= self.config.strict_entry_rejects
            ):
                rs.state = GuardState.STRICT
                rs.capture_rejects.clear()
                logger.warning(
                    f"[AmbientGuard] {room_id}: NORMAL → STRICT "
                    f"({self.config.strict_entry_rejects} capturas rechazadas en "
                    f"{self.config.strict_entry_window_s:.0f}s — ambiente hostil; "
                    f"wake ahora exige score ≥ {self.config.strict_wake_score})"
                )
            elif (
                rs.state is GuardState.STRICT
                and len(rs.capture_rejects) >= self.config.cooldown_entry_rejects
            ):
                rs.state = GuardState.COOLDOWN
                rs.cooldown_until = now + self.config.cooldown_duration_s
                rs.capture_rejects.clear()
                logger.warning(
                    f"[AmbientGuard] {room_id}: STRICT → COOLDOWN "
                    f"{self.config.cooldown_duration_s:.0f}s (rechazos persisten — "
                    f"breaker para no saturar el router)"
                )

    def state_for(self, room_id: str) -> GuardState:
        if not self.config.enabled:
            return GuardState.NORMAL
        with self._lock:
            rs = self._room(room_id)
            self._refresh(rs, self._time())
            return rs.state

    def follow_up_allowed(self, room_id: str) -> bool:
        """follow_up solo en NORMAL — en STRICT la ventana abierta era parte
        de la cascada de saturación del 06-04."""
        if not self.config.enabled:
            return True
        return self.state_for(room_id) is GuardState.NORMAL

    # ---- internos ----

    def _room(self, room_id: str) -> _RoomState:
        rs = self._rooms.get(room_id)
        if rs is None:
            rs = _RoomState()
            self._rooms[room_id] = rs
        return rs

    def _refresh(self, rs: _RoomState, now: float) -> None:
        """Transiciones por tiempo (lazy — no hay timers)."""
        if rs.state is GuardState.COOLDOWN and now >= rs.cooldown_until:
            rs.state = GuardState.STRICT
            rs.capture_rejects.clear()
            logger.info("[AmbientGuard] COOLDOWN expirado → STRICT")
        if (
            rs.state is GuardState.STRICT
            and rs.last_reject_at > 0.0
            and (now - rs.last_reject_at) >= self.config.strict_exit_quiet_s
        ):
            rs.state = GuardState.NORMAL
            rs.capture_rejects.clear()
            logger.info(
                f"[AmbientGuard] STRICT → NORMAL "
                f"(quiet ≥ {self.config.strict_exit_quiet_s:.0f}s)"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_ambient_guard.py -v`
Expected: PASS (todos — ~25 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/ambient_guard.py tests/unit/pipeline/test_ambient_guard.py
git commit -m "feat(pipeline): AmbientGuard — escalera NORMAL/STRICT/COOLDOWN por room + classify_outcome"
```

---

### Task 4: Integración en MultiRoomAudioLoop

**Files:**
- Modify: `src/pipeline/command_event.py` (campo `ambient_strict`)
- Modify: `src/pipeline/multi_room_audio_loop.py`
- Test: `tests/unit/pipeline/test_multi_room_audio_loop.py` (clase nueva al final)

Puntos de integración (verificados en el código actual):
1. `_should_accept_wakeword()` (línea ~224) gana parámetro `wake_score` y consulta el guard PRIMERO.
2. Callsite del wake en `_make_audio_callback` (línea ~520): pasa `detection[1]` como score.
3. `self.follow_up.start_conversation()` post-wake (línea ~532) y la rama follow-up sin wake (línea ~574): condicionadas a `follow_up_allowed()`. El `start_conversation()` de `_trigger_barge_in` NO se toca (barge-in implica interacción real con TTS activo).
4. `_dispatch_command()` (línea ~729): clasifica el result y reporta `on_capture_result()`.
5. Los dos `CommandEvent(...)` de `run()` (early dispatch línea ~404 y VAD normal línea ~424) setean `ambient_strict`.

- [ ] **Step 1: Write the failing tests**

Agregar al final de `tests/unit/pipeline/test_multi_room_audio_loop.py`:

```python
# ============================================================
# AmbientGuard integration (spec 2026-06-05)
# ============================================================

from src.pipeline.ambient_guard import (
    AmbientGuard,
    AmbientGuardConfig,
    GuardState,
)


def _make_enabled_guard(**overrides) -> AmbientGuard:
    cfg = AmbientGuardConfig(
        enabled=True,
        strict_entry_rejects=2,
        strict_entry_window_s=60.0,
        strict_wake_score=0.65,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return AmbientGuard(config=cfg)


class TestAmbientGuardIntegration:
    def test_no_guard_keeps_current_behavior(self):
        loop = _make_multi_room_loop()
        assert loop._should_accept_wakeword("cocina", rms=0.05, timestamp=time.time(),
                                            wake_score=0.41) is True

    def test_guard_rejects_low_score_in_strict(self):
        guard = _make_enabled_guard()
        guard.on_capture_result("cocina", "noise")
        guard.on_capture_result("cocina", "noise")  # → STRICT
        loop = _make_multi_room_loop(ambient_guard=guard)
        assert loop._should_accept_wakeword("cocina", rms=0.05, timestamp=time.time(),
                                            wake_score=0.50) is False

    def test_guard_accepts_high_score_in_strict(self):
        guard = _make_enabled_guard()
        guard.on_capture_result("cocina", "noise")
        guard.on_capture_result("cocina", "noise")
        loop = _make_multi_room_loop(ambient_guard=guard)
        assert loop._should_accept_wakeword("cocina", rms=0.05, timestamp=time.time(),
                                            wake_score=0.80) is True

    @pytest.mark.asyncio
    async def test_dispatch_reports_outcome_to_guard(self):
        guard = _make_enabled_guard()
        loop = _make_multi_room_loop(ambient_guard=guard)
        # Callback que simula rechazo del gate (texto ruido)
        loop.on_command(AsyncMock(return_value={
            "success": False, "text": "gracias por ver", "intent": "gate_rejected",
        }))
        event = CommandEvent(audio=np.zeros(16000, dtype=np.float32), room_id="cocina")
        await loop._dispatch_command(event)
        await loop._dispatch_command(event)
        # 2 rechazos con strict_entry_rejects=2 → STRICT
        assert guard.state_for("cocina") is GuardState.STRICT

    @pytest.mark.asyncio
    async def test_dispatch_accepted_does_not_escalate(self):
        guard = _make_enabled_guard()
        loop = _make_multi_room_loop(ambient_guard=guard)
        loop.on_command(AsyncMock(return_value={
            "success": True, "text": "prende la luz", "intent": "domotics",
        }))
        event = CommandEvent(audio=np.zeros(16000, dtype=np.float32), room_id="cocina")
        for _ in range(5):
            await loop._dispatch_command(event)
        assert guard.state_for("cocina") is GuardState.NORMAL

    def test_command_event_carries_ambient_strict_default_false(self):
        event = CommandEvent(audio=np.zeros(10, dtype=np.float32), room_id="cocina")
        assert event.ambient_strict is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py::TestAmbientGuardIntegration -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'ambient_guard'` (y/o `wake_score`, `ambient_strict`)

- [ ] **Step 3: Implement — CommandEvent**

En `src/pipeline/command_event.py`, agregar al final de la dataclass (después de `wake_text`):

```python
    # AmbientGuard (spec 2026-06-05): True cuando la room estaba en STRICT al
    # despachar. El RequestRouter lo usa para NO otorgar el bonus de
    # wake_acoustically_confirmed al grammar fast-path — con TV de fondo el
    # wake espurio invalida la premisa "wake ⇒ usuario".
    ambient_strict: bool = False
```

- [ ] **Step 4: Implement — MultiRoomAudioLoop**

En `src/pipeline/multi_room_audio_loop.py`:

(a) Import (junto a los demás `from src...`):

```python
from src.pipeline.ambient_guard import AmbientGuard, GuardState, classify_outcome
```

(b) Constructor: agregar parámetro `ambient_guard: AmbientGuard | None = None` (después de `xvf_tuning`) y en el body:

```python
        # AmbientGuard (spec 2026-06-05): compuerta acústica integral por room.
        # None = sin guard (comportamiento previo exacto). La escalera
        # NORMAL/STRICT/COOLDOWN se alimenta de los resultados de captura
        # (_dispatch_command) y decide sobre cada wake (_should_accept_wakeword).
        self._guard = ambient_guard
```

(c) `_should_accept_wakeword`: cambiar la firma y agregar el check del guard AL PRINCIPIO (antes del check de `min_wake_rms`):

```python
    def _should_accept_wakeword(
        self, room_id: str, rms: float, timestamp: float, wake_score: float = 1.0
    ) -> bool:
```

```python
        # AmbientGuard primero: en COOLDOWN rechaza barato (sin tocar dedup);
        # en STRICT exige score alto. El detector queda en su threshold base —
        # la decisión adaptativa vive acá, en un solo lugar testeable.
        if self._guard is not None:
            decision = self._guard.on_wake(room_id, wake_score, rms)
            if not decision.accept:
                logger.info(
                    f"[AmbientGuard] wake rechazado en {room_id} "
                    f"({decision.reason}, state={decision.state.value}, "
                    f"score={wake_score:.2f}, rms={rms:.4f})"
                )
                return False
```

(d) Callsite en `_make_audio_callback` (línea ~520), pasar el score:

```python
                if detection:
                    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                    if self._should_accept_wakeword(
                        rs.room_id, rms, time.time(), wake_score=detection[1]
                    ):
```

(e) follow_up post-wake (línea ~532) — reemplazar `self.follow_up.start_conversation()` por:

```python
                        # En STRICT/COOLDOWN no abrir follow_up: la ventana
                        # abierta con TV era parte de la cascada del 06-04.
                        if self._guard is None or self._guard.follow_up_allowed(rs.room_id):
                            self.follow_up.start_conversation()
```

(f) Rama follow-up sin wake (línea ~574) — agregar la condición del guard:

```python
                elif self.follow_up.is_active and (
                    self._guard is None or self._guard.follow_up_allowed(rs.room_id)
                ):
```

(g) Los dos constructores de `CommandEvent` en `run()` (early dispatch y VAD normal): agregar a ambos el kwarg:

```python
                            ambient_strict=(
                                self._guard is not None
                                and self._guard.state_for(room_id) is GuardState.STRICT
                            ),
```

(h) `_dispatch_command`: reportar el outcome después de obtener `result`:

```python
    async def _dispatch_command(self, event: CommandEvent):
        """Dispatch a captured command via registered callback."""
        try:
            if self._on_command_callback:
                result = await self._on_command_callback(event)
            else:
                logger.warning("No on_command callback registered")
                result = {}

            # AmbientGuard: el resultado de la captura alimenta la escalera
            # (noise/empty/timeout escalan; accepted/other_fail no).
            if self._guard is not None and isinstance(result, dict):
                outcome = classify_outcome(result)
                self._guard.on_capture_result(event.room_id, outcome)
                logger.debug(
                    f"[AmbientGuard] capture outcome en {event.room_id}: {outcome}"
                )

            if self._on_post_command_callback:
                await self._on_post_command_callback(result, event)
        except Exception as e:
            logger.exception(f"Command dispatch failed for {event.room_id}: {e}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py -v`
Expected: PASS (los preexistentes + los 6 nuevos). Si algún preexistente rompe por la firma de `_should_accept_wakeword`, el default `wake_score=1.0` debe mantenerlos verdes — revisar antes de tocar tests viejos.

- [ ] **Step 6: Correr también los tests vecinos del pipeline (aislados)**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_ambient_guard.py tests/unit/pipeline/test_multi_room_audio_loop.py tests/unit/pipeline/test_endpointing.py tests/unit/pipeline/test_barge_in.py -v`
Expected: PASS (nota: `test_endpointing::test_voice_prob_uses_vad` es una falla preexistente del baseline en laptop — si es LA ÚNICA falla, OK)

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/command_event.py src/pipeline/multi_room_audio_loop.py tests/unit/pipeline/test_multi_room_audio_loop.py
git commit -m "feat(pipeline): AmbientGuard cableado en MultiRoomAudioLoop — wake gating, follow_up, outcomes"
```

---

### Task 5: STRICT desactiva el bonus wake en RequestRouter

**Files:**
- Modify: `src/pipeline/request_router.py`
- Test: `tests/unit/pipeline/test_request_router_grammar.py`

Hoy `wake_confirmed=self.wake_acoustically_confirmed` (flag global del router, línea ~460) otorga +0.15 al grammar fast-path para textos ≤6 palabras sin "nexa" transcripta. En STRICT esa premisa ("wake ⇒ usuario") es falsa — el wake espurio de la TV la invalida. El flag `ambient_strict` del evento la apaga per-request.

Nota: solo el path orchestrated llama a `_grammar_fastpath_classification` (el legacy no — verificado). `orchestrator.enabled: true` en prod.

- [ ] **Step 1: Write the failing test**

Agregar a `tests/unit/pipeline/test_request_router_grammar.py` (seguir los fixtures/mocks existentes del archivo para construir el router; el patrón exacto está en los tests vecinos de ese archivo):

```python
class TestAmbientStrictDisablesWakeBonus:
    """En STRICT (CommandEvent.ambient_strict=True) el bonus de
    wake_acoustically_confirmed NO se otorga: 'prende la luz' (conf 0.7 <
    0.75 sin bonus) debe caer al LLMRouter en vez de ganar el fast-path."""

    @pytest.mark.asyncio
    async def test_strict_event_does_not_get_wake_bonus(self, router_with_grammar):
        # router_with_grammar: fixture existente del archivo con
        # wake_acoustically_confirmed=True y confidence_threshold=0.75.
        router = router_with_grammar
        event = make_command_event(text="prende la luz", ambient_strict=True)
        # Mock del LLM router para detectar el fall-through
        router.llm_command_router = AsyncMock()
        router.llm_command_router.classify = AsyncMock(
            return_value=make_noise_classification()
        )
        await router.process_command(event)
        # Sin bonus → grammar no alcanza 0.75 → SÍ se llama al LLM router
        router.llm_command_router.classify.assert_called_once()

    @pytest.mark.asyncio
    async def test_normal_event_keeps_wake_bonus(self, router_with_grammar):
        router = router_with_grammar
        event = make_command_event(text="prende la luz", ambient_strict=False)
        router.llm_command_router = AsyncMock()
        router.llm_command_router.classify = AsyncMock()
        await router.process_command(event)
        # Con bonus → grammar fast-path gana → LLM router NO se llama
        router.llm_command_router.classify.assert_not_called()
```

(Adaptar `router_with_grammar` / `make_command_event` / `make_noise_classification` a los helpers reales del archivo — si no existen, crearlos siguiendo el patrón de los tests vecinos que ya construyen el router con `wake_acoustically_confirmed=True` y un `command_processor` mockeado que devuelve el texto fijo.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_request_router_grammar.py -v -k ambient_strict`
Expected: FAIL — el bonus se otorga igual (classify no llamado en el primer test)

- [ ] **Step 3: Implement**

En `src/pipeline/request_router.py`:

(a) En `process_command()` (línea ~359), extraer el flag del evento:

```python
        ambient_strict = False
        if isinstance(audio_or_event, CommandEvent):
            ...  # bloque existente
            ambient_strict = getattr(audio_or_event, "ambient_strict", False)
```

(b) Pasarlo a `_process_command_orchestrated(...)` (línea ~401) como kwarg `ambient_strict=ambient_strict`, y agregar el parámetro a la firma con default `False`. (El path legacy no usa el grammar fast-path — no necesita el flag.)

(c) En la llamada al fast-path (línea ~458):

```python
        grammar_cls = _grammar_fastpath_classification(
            text, self.confidence_threshold,
            # En STRICT (AmbientGuard) el wake acústico no es evidencia de
            # usuario — la TV dispara el wake espurio. Sin bonus, el texto
            # tiene que valerse solo (o traer "nexa" transcripta).
            wake_confirmed=self.wake_acoustically_confirmed and not ambient_strict,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_request_router_grammar.py -v`
Expected: PASS (preexistentes + 2 nuevos)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/request_router.py tests/unit/pipeline/test_request_router_grammar.py
git commit -m "feat(router): ambient_strict apaga el bonus wake_acoustically_confirmed del grammar fast-path"
```

---

### Task 6: Config en settings.yaml + wiring en main.py

**Files:**
- Modify: `config/settings.yaml` (bloque nuevo bajo `rooms:`, junto a `wake_word:`)
- Modify: `src/main.py` (líneas ~826-887, construcción del `MultiRoomAudioLoop`)

- [ ] **Step 1: Agregar el bloque de config**

En `config/settings.yaml`, dentro de `rooms:` (mismo nivel que `wake_word:` — buscar la línea `endpointing:` y agregar ANTES):

```yaml
  # AmbientGuard — compuerta acústica integral (spec 2026-06-05). Escalera
  # NORMAL → STRICT (TV-mode) → COOLDOWN (breaker) POR ROOM, alimentada por la
  # tasa de capturas rechazadas (noise/vacío/timeout del router) — señal de
  # SOFTWARE que no depende del chip (funciona igual con el mic UAC1.0).
  # En STRICT: wake exige score ≥ strict_wake_score (el threshold base 0.40
  # del detector NO cambia), follow_up off, y el grammar fast-path pierde el
  # bonus de wake acústico (se exige "nexa" en el texto). En COOLDOWN se
  # descarta todo por cooldown_duration_s → la cola del router NUNCA se
  # satura ("no me escucha" del 06-04/05 = comandos compitiendo en la cola).
  # ⚠️ Umbrales acústicos (strict_wake_score/strict_min_rms/strict_min_spenergy):
  # calibrar con tools/acoustic_calibration.py — NO adivinar.
  # enabled: false = guard 100% pasivo (comportamiento actual exacto).
  ambient_guard:
    enabled: false
    strict_entry_rejects: 4        # capturas rechazadas en la ventana → STRICT
    strict_entry_window_s: 60.0
    strict_exit_quiet_s: 120.0     # histéresis: quiet sostenido → NORMAL
    strict_wake_score: 0.65        # ⚠️ placeholder — calibrar con la matriz
    strict_min_rms: 0.0            # 0.0 = no aplicar (calibrar o dejar muerto)
    strict_min_spenergy: 0.0       # 0.0 = no aplicar (calibrar o dejar muerto)
    cooldown_entry_rejects: 6      # rechazos MÁS estando en STRICT → COOLDOWN
    cooldown_entry_window_s: 60.0
    cooldown_duration_s: 30.0
```

- [ ] **Step 2: Wiring en main.py**

En `src/main.py`, antes de `multi_room_loop = MultiRoomAudioLoop(` (línea ~857):

```python
            # AmbientGuard (spec 2026-06-05): compuerta acústica integral.
            # Default enabled=false → pasivo. Umbrales: ver settings.yaml.
            from src.pipeline.ambient_guard import AmbientGuard, AmbientGuardConfig
            ag_cfg = rooms_config.get("ambient_guard", {}) or {}
            ambient_guard = AmbientGuard(
                config=AmbientGuardConfig(
                    enabled=ag_cfg.get("enabled", False),
                    strict_entry_rejects=ag_cfg.get("strict_entry_rejects", 4),
                    strict_entry_window_s=ag_cfg.get("strict_entry_window_s", 60.0),
                    strict_exit_quiet_s=ag_cfg.get("strict_exit_quiet_s", 120.0),
                    strict_wake_score=ag_cfg.get("strict_wake_score", 0.65),
                    strict_min_rms=ag_cfg.get("strict_min_rms", 0.0),
                    strict_min_spenergy=ag_cfg.get("strict_min_spenergy", 0.0),
                    cooldown_entry_rejects=ag_cfg.get("cooldown_entry_rejects", 6),
                    cooldown_entry_window_s=ag_cfg.get("cooldown_entry_window_s", 60.0),
                    cooldown_duration_s=ag_cfg.get("cooldown_duration_s", 30.0),
                )
            )
            if ag_cfg.get("enabled", False):
                logger.info(
                    f"AmbientGuard ACTIVO: strict_wake_score="
                    f"{ag_cfg.get('strict_wake_score', 0.65)}, "
                    f"entry={ag_cfg.get('strict_entry_rejects', 4)} rechazos/"
                    f"{ag_cfg.get('strict_entry_window_s', 60.0):.0f}s"
                )
```

Y en el constructor del loop, agregar el kwarg:

```python
                ambient_guard=ambient_guard,
```

- [ ] **Step 3: Sanity check de import del main**

Run: `.venv/bin/python -c "import ast; ast.parse(open('src/main.py').read()); import yaml; yaml.safe_load(open('config/settings.yaml')); print('OK')"`
Expected: `OK` (main.py no se puede importar en laptop sin hardware; validamos sintaxis + yaml)

- [ ] **Step 4: Commit**

```bash
git add config/settings.yaml src/main.py
git commit -m "feat(config): bloque rooms.ambient_guard (pasivo por default) + wiring en main"
```

---

### Task 7: Deploy del harness y del guard (pasivo) al server

Manual con el usuario presente. El guard va con `enabled: false` — este deploy NO cambia comportamiento, solo lleva el código y el harness.

- [ ] **Step 1: Push desde la laptop**

```bash
scripts/kza-push   # o: git push origin feat/nexa-command-detection-fixes
```

- [ ] **Step 2: Pull en el server y verificar**

```bash
ssh kza "cd /home/kza/app && git pull && git log --oneline -3"
```
Expected: HEAD del server == HEAD local. NO reiniciar kza-voice todavía (el guard está pasivo; el restart va junto con el flip en Task 10 — un solo restart, VRAM apretada en cuda:1: chequear `nvidia-smi` antes).

---

### Task 8: Matriz de calibración en vivo (server, usuario presente)

⚠️ Requiere: usuario en el escritorio, TV disponible, kza-voice PARADO.

- [ ] **Step 1: Parar kza-voice y localizar el device**

```bash
ssh kza "systemctl --user stop kza-voice"
ssh kza "cd /home/kza/app && .venv/bin/python -c \"import sounddevice as sd; print(sd.query_devices())\""
```
Anotar el índice del XVF3800 (en deploys previos fue ~10 para living `1-2.2`, ~4 para escritorio `3-1.1` — VERIFICAR contra la salida real, los índices cambian con re-enumeración).

- [ ] **Step 2: Correr las 4 condiciones (coordinar con el usuario por chat)**

```bash
ssh kza "cd /home/kza/app && .venv/bin/python -m tools.acoustic_calibration \
  --condition silencio --duration 120 --device <IDX> --channel 1"
# Usuario prende la TV a volumen normal:
ssh kza "cd /home/kza/app && .venv/bin/python -m tools.acoustic_calibration \
  --condition tv --duration 180 --device <IDX> --channel 1"
# TV apagada, usuario dice ~10 comandos reales con 'Nexa' a distancia normal:
ssh kza "cd /home/kza/app && .venv/bin/python -m tools.acoustic_calibration \
  --condition voz --duration 120 --device <IDX> --channel 1"
# TV prendida + ~10 comandos reales:
ssh kza "cd /home/kza/app && .venv/bin/python -m tools.acoustic_calibration \
  --condition voz_tv --duration 120 --device <IDX> --channel 1"
```

- [ ] **Step 3: Analizar**

```bash
ssh kza "cd /home/kza/app && .venv/bin/python -m tools.acoustic_calibration --analyze data/calibration"
```

- [ ] **Step 4: Reiniciar kza-voice (vuelve el servicio mientras se decide)**

```bash
ssh kza "nvidia-smi --query-gpu=memory.used,memory.free --format=csv && systemctl --user start kza-voice && sleep 20 && systemctl --user status kza-voice --no-pager | head -15"
```
Expected: servicio active, arranque limpio en logs.

- [ ] **Step 5: Copiar los JSONL a la laptop (dataset para futuro re-entrenamiento del wake)**

```bash
scp -r kza:/home/kza/app/data/calibration data/ && git add data/calibration && git status
```
(Si los JSONL son grandes >10MB, NO commitearlos — dejarlos en `data/` local y anotarlo.)

---

### Task 9: Decisión de umbrales (con el usuario — checkpoint de diseño)

- [ ] **Step 1: Presentar la tabla de gaps al usuario y decidir juntos:**

| Señal | Si separa (gap > 0) | Si NO separa |
|---|---|---|
| `wake` score voz vs tv | `strict_wake_score` = umbral recomendado (cap a ≤0.85 — más arriba mata far-field) | la escalera carga todo el peso; `strict_wake_score` queda en el p95 de tv + margen mínimo |
| `rms` voz vs tv | `strict_min_rms` = umbral recomendado | dejar 0.0 + documentar "muerto post-AGC" en el yaml |
| `spenergy` voz vs tv | `strict_min_spenergy` = umbral; evaluar también re-habilitar `spenergy_gate` global con ese umbral | dejar 0.0 + actualizar el comentario del yaml a "MUERTO con MAXGAIN=8, medido 2026-06-05" — veredicto final, deja de ser candidato |

Criterio del usuario (sesión 2026-06-05): prioridad falsos positivos — ante la duda, umbral más estricto.

---

### Task 10: Flip de config con umbrales medidos + deploy

- [ ] **Step 1: Editar `config/settings.yaml`** con los valores decididos en Task 9: `ambient_guard.enabled: true` + umbrales medidos + comentario con fecha y referencia a los JSONL. Si SPENERGY revivió: `spenergy_gate.enabled: true` + threshold medido + actualizar el comentario del bloque.

- [ ] **Step 2: Tests rápidos de regresión local**

```bash
.venv/bin/python -m pytest tests/unit/pipeline/test_ambient_guard.py tests/unit/pipeline/test_multi_room_audio_loop.py tests/unit/pipeline/test_request_router_grammar.py tests/unit/tools/test_acoustic_calibration.py -q
```
Expected: PASS

- [ ] **Step 3: Commit + push + deploy + restart (con OK del usuario)**

```bash
git add config/settings.yaml
git commit -m "feat(config): AmbientGuard ON con umbrales calibrados 2026-06-05 (matriz voz/tv en vivo)"
scripts/kza-push
ssh kza "cd /home/kza/app && git pull && nvidia-smi --query-gpu=memory.used,memory.free --format=csv"
# Pedir OK explícito del usuario antes del restart (producción del hogar):
ssh kza "systemctl --user restart kza-voice && sleep 25 && journalctl --user -u kza-voice --since '1 min ago' --no-pager | grep -E 'AmbientGuard|ready|ERROR' | head"
```
Expected: log `AmbientGuard ACTIVO: ...`, servicio ready, 0 errores.

---

### Task 11: Validación en vivo (protocolo del spec)

- [ ] **Step 1: TV 10 min sin hablar** — monitorear: `ssh kza "journalctl --user -u kza-voice -f | grep -E 'AmbientGuard|LLMRouter|timeout'"`. Esperado: el guard escala a STRICT y se queda (sin oscilar); 0 acciones; 0 timeouts del router en cadena.
- [ ] **Step 2: Comando con TV (voz firme)** — esperado: funciona, o rechazo honesto y al repetir más cerca funciona.
- [ ] **Step 3: TV apagada + quiet ~2-3 min (según strict_exit_quiet_s)** — esperado: log `STRICT → NORMAL`; comandos 3/3 (paridad con la ronda 4 del 06-04).
- [ ] **Step 4: Prender TV de nuevo en NORMAL** — esperado: re-escala a STRICT sin acción fantasma en la transición.
- [ ] **Step 5: Si todo pasa** — actualizar la memoria del proyecto (archivo de sesión XVF3800) con el resultado + actualizar el spec con los umbrales finales. Si sobra sesión: A/B `AEC_ASROUTGAIN` 1→2 (fuera del core).

---

## Self-Review (hecho al escribir el plan)

- **Cobertura del spec:** harness (T1-2), AmbientGuard (T3), integración loop+follow_up+outcomes (T4), bonus wake en STRICT (T5), config+DI (T6), matriz en vivo (T7-8), decisión por datos (T9), flip+deploy (T10), protocolo de validación (T11). Manejo de errores: fail-open USB (spenergy_peak=None test en T3), living sin chip (señal de software únicamente), defaults pasivos (T3 TestPassiveDefaults, T6 enabled:false). ✔
- **Sin placeholders:** todo step de código tiene el código; el único valor "placeholder" es `strict_wake_score: 0.65` en el yaml, marcado explícitamente como a-calibrar y con el guard `enabled: false` hasta Task 10. ✔
- **Consistencia de tipos:** `on_wake(room_id, score, rms, spenergy_peak=None) -> GuardDecision`, `on_capture_result(room_id, outcome)`, `state_for(room_id) -> GuardState`, `follow_up_allowed(room_id) -> bool`, `classify_outcome(result) -> str` usados idénticos en T3/T4/T5/T6. `GuardState` importado en multi_room_audio_loop para `ambient_strict`. ✔
