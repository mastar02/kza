# Transcripción Continua Multi-Pista (Ambient Path) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ambient path en `cuda:0` que transcribe continuamente las pistas del XVF3800 (firmware 6ch), etiqueta cada utterance con hablante/dirección/fuente (live|tv|self|unknown), persiste con TTL corto y destila hechos a memoria con LLM local.

**Architecture:** El command path (cuda:1, <300ms) no se toca. Un `MultiChannelTap` se cuelga del audio callback existente de `MultiRoomAudioLoop` y alimenta un pipeline asíncrono best-effort: Silero VAD sobre mic crudo segmenta utterances → Whisper turbo (instancia separada en cuda:0) transcribe el beam ASR → ECAPA etiqueta hablante → GCC-PHAT da dirección → clasificador marca fuente → SQLite con TTL → destilador periódico vía Qwen 7B local (:8101) → ChromaDB memoria.

**Tech Stack:** Python 3.13, sounddevice, faster-whisper, Silero VAD (torch.hub, patrón existente en `whisper_wake.py:570`), speechbrain ECAPA, numpy (GCC-PHAT), aiosqlite, aiohttp.

**Spec:** `docs/superpowers/specs/2026-06-06-transcripcion-continua-multipista-design.md`

**Branch de trabajo:** crear `feat/ambient-multipista` desde `feat/nexa-command-detection-fixes` (el spec ya está commiteado ahí):
```bash
git checkout -b feat/ambient-multipista
```

**Comando de test (laptop):** `.venv/bin/python -m pytest tests/unit/ambient/ -v` — ⚠️ usar SIEMPRE `.venv/bin/python` (el python3 del sistema es 3.9 y rompe con `slots=True`). Correr `tests/unit/ambient/` aislado, igual que vectordb/grammar_fastpath (mocks de `sys.modules` de otros módulos interfieren en la suite completa).

**Desviaciones registradas respecto del spec** (decididas al planificar, mantienen la intención):
1. `source` gana el valor `self` (audio capturado mientras nuestro TTS hablaba) — evita que el RAG memorice lo que dice el propio asistente. El tap propaga `tts_active` desde `response_handler.is_speaking` que `MultiRoomAudioLoop` ya conoce.
2. `tv_azimuth` se calibra en azimut *relativo* (atan2 sobre TDOAs GCC-PHAT de pares de mics) — discrimina dirección-TV sin necesitar la geometría absoluta del array. Misma config, misma semántica.
3. `ambient.stt` hereda del bloque `stt:` top-level y overridea `device=cuda:0`, `beam_size=1`, `initial_prompt=None` (el prompt domótico causaba alucinaciones copiándolo literal — visto en wake).
4. Sin enrolamiento de voces (estado actual: `data/users.json` no existe), la regla "hablante conocido → live" no puede disparar; el discriminador principal es DoA-no-TV. `require_known_speaker_for_live: false` (default) refleja esto.

---

## File Structure

```
src/ambient/                      (ya existe; hoy solo audio_event_detector.py)
├── types.py                      Create — AmbientUtterance, RawSegment (DTOs)
├── tap.py                        Create — MultiChannelTap (ring buffers thread-safe)
├── segmenter.py                  Create — UtteranceSegmenter + factory Silero
├── doa.py                        Create — gcc_phat() + DoAEstimator
├── ambient_stt.py                Create — AmbientSTT (wrapper async de FastWhisperSTT)
├── speaker_tagger.py             Create — SpeakerTagger (ECAPA → speaker, conf)
├── source_classifier.py          Create — SourceClassifier (live|tv|self|unknown)
├── store.py                      Create — AmbientStore (aiosqlite, TTL)
├── distiller.py                  Create — Distiller (LLM local → LongTermMemory)
└── transcriber.py                Create — AmbientTranscriber (orquestador) + build_ambient_path()

src/pipeline/multi_room_audio_loop.py   Modify — attach_ambient(), tee en callback, shadow anti-TV
src/main.py                              Modify — wiring ambient (post multi_room_loop)
config/settings.yaml                     Modify — sección ambient: (al final)
tools/benchmark_ambient.py               Create — Fase 0
docs/runbooks/2026-06-06-xvf3800-flasheo-6ch.md  Create — Fase 1 (ejecución GATED)
tests/unit/ambient/                      Create — un test file por componente
```

Mapa fases del spec → tareas: Fase 0 = Tasks 1-2 · Fase 1 = ejecución del runbook (GATED, fuera del plan de código) · Fase 2 = Tasks 3-14 (shadow) · Fase 3 = flip de config con datos (fuera del plan de código).

---

### Task 1: Benchmark Fase 0 (`tools/benchmark_ambient.py`)

Script standalone para medir en el server (cuda:0) lo que el spec exige antes de comprometer: RTF/VRAM del STT continuo, throughput Silero y GCC-PHAT. Modo `--smoke` corre en laptop sin GPU con audio sintético. Parakeet v3 queda como opcional explícito: si Whisper turbo da RTF < 0.2 con < 2 GB, YAGNI.

**Files:**
- Create: `tools/benchmark_ambient.py`

- [ ] **Step 1: Escribir el script completo**

```python
"""Benchmark Fase 0 — transcripción continua multi-pista (spec 2026-06-06).

Mide en el server (cuda:0) los números que el diseño exige antes de comprometer:
  --stt    RTF y VRAM de faster-whisper turbo sobre segmentos de voz
  --vad    throughput de Silero VAD (CPU) en ventanas de 512 samples
  --doa    throughput de GCC-PHAT (CPU) sobre pares de canales
  --smoke  todo lo anterior con audio sintético corto y device=cpu (laptop)

Uso (server):
  .venv/bin/python tools/benchmark_ambient.py --stt --device cuda:0 \
      --model /home/kza/kza/models/whisper-v3-turbo --wav sample.wav
  .venv/bin/python tools/benchmark_ambient.py --vad --doa
Uso (laptop): .venv/bin/python tools/benchmark_ambient.py --smoke
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("benchmark_ambient")

SAMPLE_RATE = 16000


def synth_speech(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Pseudo-voz sintética (tonos modulados) — suficiente para medir throughput.

    NO sirve para WER (eso se mide con --wav y audio real grabado del mic).
    """
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    f0 = 140 + 40 * np.sin(2 * np.pi * 0.7 * t)
    sig = 0.3 * np.sin(2 * np.pi * f0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3.1 * t))
    return sig.astype(np.float32)


def bench_stt(model: str, device: str, wav: str | None, iterations: int) -> None:
    import torch  # torch PRIMERO (regla orden imports CUDA del proyecto)
    from src.stt.whisper_fast import FastWhisperSTT

    if wav:
        import soundfile as sf
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != SAMPLE_RATE:
            raise SystemExit(f"wav debe ser 16kHz (es {sr})")
    else:
        audio = synth_speech(8.0)
        logger.info("⚠️  audio sintético: RTF válido, WER no medible")

    stt = FastWhisperSTT(
        model=model, device=device, compute_type="int8_float16",
        language="es", beam_size=1, initial_prompt=None, vad_filter=False,
    )
    stt.load()
    is_cuda = "cuda" in device
    if is_cuda:
        dev_idx = int(device.split(":")[-1]) if ":" in device else 0
        torch.cuda.synchronize(dev_idx)
        vram_before = torch.cuda.memory_allocated(dev_idx)

    # warmup
    stt.transcribe(audio[: SAMPLE_RATE * 2])

    dur_s = len(audio) / SAMPLE_RATE
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        text, _ = stt.transcribe(audio)
        times.append(time.perf_counter() - t0)
    rtf = float(np.median(times)) / dur_s
    logger.info(f"STT {model} @ {device}")
    logger.info(f"  audio={dur_s:.1f}s iter={iterations}")
    logger.info(f"  latencia mediana={np.median(times)*1000:.0f}ms p95={np.percentile(times,95)*1000:.0f}ms")
    logger.info(f"  RTF={rtf:.3f}  (continuo viable si <0.5; cómodo si <0.2)")
    logger.info(f"  texto: {text[:80]!r}")
    if is_cuda:
        torch.cuda.synchronize(dev_idx)
        logger.info(f"  VRAM torch allocated: {torch.cuda.memory_allocated(dev_idx)/2**20:.0f} MiB "
                    f"(delta carga: ver nvidia-smi — ctranslate2 no reporta a torch)")
        logger.info("  ⚠️ medir VRAM real con: nvidia-smi --query-compute-apps=pid,used_memory --format=csv")


def bench_vad(iterations: int) -> None:
    from src.ambient.segmenter import make_silero_predictor

    predict = make_silero_predictor()
    chunk = synth_speech(0.08)  # 1280 samples = chunk del pipeline
    predict(chunk)  # warmup
    t0 = time.perf_counter()
    for _ in range(iterations):
        predict(chunk)
    per_chunk_ms = (time.perf_counter() - t0) / iterations * 1000
    logger.info(f"Silero VAD: {per_chunk_ms:.2f} ms/chunk de 80ms "
                f"({per_chunk_ms/80*100:.1f}% de un core — continuo viable si <50%)")


def bench_doa(iterations: int) -> None:
    from src.ambient.doa import DoAEstimator

    est = DoAEstimator()
    seg = np.random.default_rng(7).normal(0, 0.1, size=(SAMPLE_RATE * 3, 6)).astype(np.float32)
    est.estimate(seg)  # warmup
    t0 = time.perf_counter()
    for _ in range(iterations):
        est.estimate(seg)
    per_seg_ms = (time.perf_counter() - t0) / iterations * 1000
    logger.info(f"DoA GCC-PHAT: {per_seg_ms:.1f} ms por segmento de 3s (CPU)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stt", action="store_true")
    ap.add_argument("--vad", action="store_true")
    ap.add_argument("--doa", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="todo en CPU con audio corto (laptop)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="./models/whisper-v3-turbo")
    ap.add_argument("--wav", default=None, help="audio real 16kHz mono para RTF/WER")
    ap.add_argument("--iterations", type=int, default=10)
    args = ap.parse_args()

    if args.smoke:
        bench_doa(3)
        bench_vad(10)
        bench_stt(model=args.model, device="cpu", wav=args.wav, iterations=2)
        return
    if not (args.stt or args.vad or args.doa):
        ap.error("elegí al menos uno: --stt/--vad/--doa/--smoke")
    if args.doa:
        bench_doa(args.iterations)
    if args.vad:
        bench_vad(args.iterations)
    if args.stt:
        bench_stt(model=args.model, device=args.device, wav=args.wav, iterations=args.iterations)


if __name__ == "__main__":
    main()
```

Nota: `--vad`/`--doa` importan de `src.ambient.segmenter`/`src.ambient.doa` que se crean en Tasks 5-6 — el script se commitea ahora (Fase 0 puede correr `--stt` ya mismo; `--vad`/`--doa` corren cuando esas tasks estén). `--smoke` completo recién post-Task 6.

- [ ] **Step 2: Verificar sintaxis**

Run: `.venv/bin/python -m py_compile tools/benchmark_ambient.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tools/benchmark_ambient.py
git commit -m "feat(ambient): benchmark Fase 0 — RTF/VRAM STT, Silero, GCC-PHAT"
```

---

### Task 2: Runbook flasheo firmware 6ch (Fase 1 — ejecución GATED)

Documento operacional. **La ejecución requiere OK explícito del usuario en el momento** (único mic de producción del hogar, en `escritorio`, puerto USB `3-1.4`).

**Files:**
- Create: `docs/runbooks/2026-06-06-xvf3800-flasheo-6ch.md`

- [ ] **Step 1: Escribir el runbook**

````markdown
# Runbook — Flasheo XVF3800 a firmware USB 6 canales

**⚠️ GATED: pedir OK explícito al usuario antes de ejecutar. Único mic de
producción (escritorio, puerto USB 3-1.4). Ventana de bajo uso del hogar.**

## Qué cambia
- Firmware actual: USB 2ch (ch0=Conference, ch1=ASR), familia 2.0.x (VERSION leída 2.0.6).
- Firmware destino: `respeaker_xvf3800_usb_dfu_firmware_6chl_v2.0.x.bin` (Seeed):
  ch0=Conference, ch1=ASR, ch2-5=mics 0-3 crudos. 16 kHz / 32-bit, UAC2 estándar.
- El binding por puerto físico (`mic_usb_port: "3-1.4"`) NO cambia.
- `capture_channel: 1` (ASR) conserva su semántica — el command path no nota el cambio.
- El tuning RAM (PP_AGCMAXGAIN=8) se pierde con el reset del flasheo;
  `xvf_tuning.apply_on_start: true` lo re-aplica al reiniciar kza-voice.

## Pre-requisitos (hacer ANTES de pedir la ventana)
1. Descargar AMBOS binarios (6ch destino + 2ch actual para rollback) desde el
   wiki de Seeed (https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/,
   sección DFU/firmware) a `/home/kza/firmware/`. Verificar que la versión 2.0.x
   coincida entre ambos.
2. `ssh kza "which dfu-util || sudo apt install dfu-util"`
3. Snapshot del chip:
   `ssh kza "cd ~/app && .venv/bin/python -m tools.xvf_tune --read-all > /home/kza/firmware/xvf_snapshot_pre6ch.txt"`

## Procedimiento
```bash
ssh kza
systemctl --user stop kza-voice
lsusb | grep 2886                      # confirmar VID 2886 presente
dfu-util -l                            # listar; confirmar XVF3800 en modo runtime
dfu-util -R -e -a 1 -D /home/kza/firmware/respeaker_xvf3800_usb_dfu_firmware_6chl_v2.0.x.bin
sleep 5
```

## Verificación (todas deben pasar)
```bash
# 1. Re-enumeró con 6 canales
arecord -l | grep -i xvf               # anotar card N
arecord -D hw:N,0 -c 6 -f S32_LE -r 16000 -d 3 /tmp/test6ch.wav && echo CAPTURA-OK
# 2. El control USB sigue vivo (xvf_host protocol)
cd ~/app && .venv/bin/python -m tools.xvf_tune --read-all | head -20
# 3. El puerto físico no cambió
readlink -f /sys/class/sound/cardN | grep "3-1.4" && echo PUERTO-OK
# 4. kza-voice levanta y abre 6 canales
systemctl --user start kza-voice
journalctl --user -u kza-voice -n 50 | grep -E "channels=|stream started"
#    esperado: "audio stream started (device=X, channels=6)"
# 5. Comando de voz real: "Nexa, prendé la luz del escritorio" → debe funcionar
```

## Rollback (si CUALQUIER verificación falla)
```bash
systemctl --user stop kza-voice
dfu-util -R -e -a 1 -D /home/kza/firmware/respeaker_xvf3800_usb_dfu_firmware_v2.0.x.bin
sleep 5
arecord -D hw:N,0 -c 2 -f S32_LE -r 16000 -d 3 /tmp/test2ch.wav && echo ROLLBACK-OK
systemctl --user start kza-voice
```
Si el device NO enumera tras un flasheo fallido: desenchufar/re-enchufar el mic
(reset por power-cycle) y reintentar dfu-util. El bootloader DFU del XVF3800 es
parte del chip — un .bin corrupto no lo brickea, pero verificar checksum antes.
````

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/2026-06-06-xvf3800-flasheo-6ch.md
git commit -m "docs(runbook): flasheo XVF3800 a firmware 6ch con verificación y rollback"
```

---

### Task 3: DTOs (`src/ambient/types.py`)

**Files:**
- Create: `src/ambient/types.py`
- Test: `tests/unit/ambient/test_types.py`

- [ ] **Step 1: Crear `tests/unit/ambient/__init__.py` vacío y escribir el failing test**

```python
"""Tests: DTOs del ambient path (spec 2026-06-06)."""
import numpy as np

from src.ambient.types import AmbientUtterance, RawSegment, SOURCE_VALUES


def test_ambient_utterance_defaults():
    u = AmbientUtterance(room_id="escritorio", t0=100.0, t1=103.5)
    assert u.text == ""
    assert u.speaker == "unknown"
    assert u.source == "unknown"
    assert u.azimuth is None
    assert u.during_tts is False
    assert u.distilled is False


def test_source_values_cerrados():
    # El clasificador y el store validan contra este set — 'self' incluido
    # (desviación 1 del plan: audio durante TTS propio no va al RAG).
    assert SOURCE_VALUES == {"live", "tv", "self", "unknown"}


def test_raw_segment_holds_multichannel_audio():
    audio = np.zeros((16000, 6), dtype=np.float32)
    seg = RawSegment(t0=1.0, t1=2.0, audio=audio, during_tts=True)
    assert seg.audio.shape == (16000, 6)
    assert seg.during_tts is True
    assert seg.duration_s == 1.0
```

- [ ] **Step 2: Run test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ambient.types'`

- [ ] **Step 3: Implementar `src/ambient/types.py`**

```python
"""DTOs del ambient path (spec 2026-06-06-transcripcion-continua-multipista).

AmbientUtterance es la unidad de salida del pipeline: un tramo de voz
transcripto y etiquetado. RawSegment es la unidad intermedia que emite el
segmentador (audio multicanal crudo, aún sin transcribir).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Valores cerrados de `source`. 'self' = capturado mientras nuestro TTS
# hablaba (no se destila a memoria — el asistente no se cita a sí mismo).
SOURCE_VALUES = {"live", "tv", "self", "unknown"}


@dataclass
class RawSegment:
    """Segmento de voz detectado por el VAD, audio multicanal sin procesar."""

    t0: float                 # epoch s — inicio (incluye pre-roll)
    t1: float                 # epoch s — fin
    audio: np.ndarray         # shape (n_samples, n_channels) float32
    during_tts: bool = False  # algún chunk llegó con el TTS propio activo

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0


@dataclass
class AmbientUtterance:
    """Utterance transcripta y etiquetada — fila de data/ambient.db."""

    room_id: str
    t0: float
    t1: float
    text: str = ""
    speaker: str = "unknown"
    speaker_confidence: float = 0.0
    azimuth: float | None = None        # rad relativo (DoA propio); None = no disponible
    azimuth_stability: float = 0.0      # 0-1, dispersión circular entre sub-ventanas
    source: str = "unknown"             # ver SOURCE_VALUES
    confidence: float | None = None     # avg_logprob del STT
    no_speech_prob: float | None = None
    during_tts: bool = False
    distilled: bool = False
```

- [ ] **Step 4: Run tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_types.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ambient/types.py tests/unit/ambient/
git commit -m "feat(ambient): DTOs AmbientUtterance y RawSegment"
```

---

### Task 4: MultiChannelTap (`src/ambient/tap.py`)

Ring buffer thread-safe por room. `push()` corre en el thread C de sounddevice — debe ser O(1), nunca lanzar, nunca bloquear. `deque(maxlen=...)` con append/popleft es thread-safe en CPython.

**Files:**
- Create: `src/ambient/tap.py`
- Test: `tests/unit/ambient/test_tap.py`

- [ ] **Step 1: Escribir el failing test**

```python
"""Tests: MultiChannelTap — ring buffer del audio callback al ambient path."""
import numpy as np

from src.ambient.tap import MultiChannelTap


def _chunk(val: float = 0.1) -> np.ndarray:
    return np.full((1280, 6), val, dtype=np.float32)


def test_push_drain_roundtrip():
    tap = MultiChannelTap(maxlen_chunks=10)
    tap.register_room("escritorio")
    tap.push("escritorio", _chunk(0.1), ts=100.0)
    tap.push("escritorio", _chunk(0.2), ts=100.08, tts_active=True)

    items = tap.drain("escritorio")
    assert len(items) == 2
    ts0, chunk0, tts0 = items[0]
    ts1, chunk1, tts1 = items[1]
    assert ts0 == 100.0 and tts0 is False
    assert ts1 == 100.08 and tts1 is True
    assert chunk1[0, 0] == np.float32(0.2)
    # drain vacía la cola
    assert tap.drain("escritorio") == []


def test_maxlen_discards_oldest_fifo():
    tap = MultiChannelTap(maxlen_chunks=3)
    tap.register_room("escritorio")
    for i in range(5):
        tap.push("escritorio", _chunk(float(i)), ts=float(i))
    items = tap.drain("escritorio")
    assert len(items) == 3
    assert [ts for ts, _, _ in items] == [2.0, 3.0, 4.0]


def test_push_unregistered_room_is_noop():
    tap = MultiChannelTap(maxlen_chunks=3)
    tap.push("living", _chunk(), ts=1.0)  # no debe lanzar
    assert tap.drain("living") == []


def test_rooms_are_independent():
    tap = MultiChannelTap(maxlen_chunks=3)
    tap.register_room("escritorio")
    tap.register_room("living")
    tap.push("escritorio", _chunk(), ts=1.0)
    assert len(tap.drain("escritorio")) == 1
    assert tap.drain("living") == []
```

- [ ] **Step 2: Run test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_tap.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ambient.tap'`

- [ ] **Step 3: Implementar `src/ambient/tap.py`**

```python
"""MultiChannelTap — puente del audio callback (thread C) al ambient path.

push() corre dentro del callback de sounddevice: O(1), sin locks bloqueantes,
sin excepciones hacia afuera. deque(maxlen) es thread-safe para append/popleft
en CPython. Si el consumidor se atrasa, se descarta FIFO (perder transcript
ambiental es aceptable; bloquear el audio del command path no).
"""
from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# ~100s de audio por room @ chunks de 80ms. A 6ch float32: ~12 MB/room.
DEFAULT_MAXLEN_CHUNKS = 1250


class MultiChannelTap:
    """Ring buffer por room de chunks multicanal (ts, chunk, tts_active)."""

    def __init__(self, maxlen_chunks: int = DEFAULT_MAXLEN_CHUNKS):
        self._maxlen = maxlen_chunks
        self._queues: dict[str, deque] = {}

    def register_room(self, room_id: str) -> None:
        """Registrar una room (idempotente)."""
        if room_id not in self._queues:
            self._queues[room_id] = deque(maxlen=self._maxlen)

    def push(
        self,
        room_id: str,
        chunk: np.ndarray,
        ts: float | None = None,
        tts_active: bool = False,
    ) -> None:
        """Encolar un chunk. Room no registrada = no-op silencioso (fail-open).

        El caller (audio callback) ya hace .copy() del buffer de PortAudio —
        acá NO se copia de nuevo.
        """
        q = self._queues.get(room_id)
        if q is None:
            return
        q.append((ts if ts is not None else time.time(), chunk, tts_active))

    def drain(self, room_id: str) -> list[tuple[float, np.ndarray, bool]]:
        """Vaciar y devolver los chunks pendientes de una room (orden FIFO)."""
        q = self._queues.get(room_id)
        if q is None:
            return []
        items = []
        while True:
            try:
                items.append(q.popleft())
            except IndexError:
                return items
```

- [ ] **Step 4: Run tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_tap.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ambient/tap.py tests/unit/ambient/test_tap.py
git commit -m "feat(ambient): MultiChannelTap — ring buffer thread-safe del callback al ambient path"
```

---

### Task 5: UtteranceSegmenter (`src/ambient/segmenter.py`)

Máquina de estados sobre Silero VAD aplicado al mic crudo (col `vad_col`, default 2 — con firmware 2ch esa columna no existe y el caller degrada a col 0). El predictor es inyectable (`vad_predict: Callable[[np.ndarray], float]`) para TDD puro; la factory real usa `torch.hub` con el patrón existente de `whisper_wake.py:570`.

**Files:**
- Create: `src/ambient/segmenter.py`
- Test: `tests/unit/ambient/test_segmenter.py`

- [ ] **Step 1: Escribir el failing test**

```python
"""Tests: UtteranceSegmenter — VAD chunked → RawSegment con pre-roll."""
import numpy as np

from src.ambient.segmenter import UtteranceSegmenter

SR = 16000
CHUNK = 1280  # 80ms


def _chunk(val: float) -> np.ndarray:
    return np.full((CHUNK, 6), val, dtype=np.float32)


def _make_segmenter(probs: list[float], **kw) -> tuple[UtteranceSegmenter, list]:
    """Segmenter con VAD fake que devuelve probs en orden (luego 0.0)."""
    seq = list(probs)
    calls: list = []

    def fake_vad(mono: np.ndarray) -> float:
        calls.append(mono.copy())
        return seq.pop(0) if seq else 0.0

    seg = UtteranceSegmenter(
        vad_predict=fake_vad, sample_rate=SR, vad_col=2,
        speech_threshold=0.5, close_silence_ms=160,  # 2 chunks de silencio
        preroll_ms=80, max_segment_s=30.0, min_speech_ms=80,
        **kw,
    )
    return seg, calls


def test_opens_and_closes_segment_with_preroll():
    # silencio, voz, voz, silencio, silencio → cierra con 2 chunks de cola
    seg, calls = _make_segmenter([0.0, 0.9, 0.9, 0.0, 0.0])
    out = []
    for i, p in enumerate([0.0, 0.1, 0.2, 0.0, 0.0]):
        out.extend(seg.feed(ts=100.0 + i * 0.08, chunk=_chunk(p)))
    assert len(out) == 1
    s = out[0]
    # pre-roll de 1 chunk (80ms): el segmento incluye el chunk de silencio previo
    assert s.t0 == 100.0
    # 5 chunks en total (preroll + 2 voz + 2 silencio de cola)
    assert s.audio.shape == (CHUNK * 5, 6)
    assert s.during_tts is False
    # el VAD recibió la columna cruda (col 2)
    assert all(c.shape == (CHUNK,) for c in calls)


def test_short_blip_below_min_speech_is_discarded():
    # 1 solo chunk de voz (80ms) con min_speech_ms=160 → descartado
    seg2 = UtteranceSegmenter(
        vad_predict=lambda m: 0.0, sample_rate=SR, vad_col=2,
        speech_threshold=0.5, close_silence_ms=160, preroll_ms=0,
        max_segment_s=30.0, min_speech_ms=160,
    )
    probs = iter([0.9, 0.0, 0.0, 0.0])
    seg2._vad_predict = lambda m: next(probs)
    out = []
    for i in range(4):
        out.extend(seg2.feed(ts=1.0 + i * 0.08, chunk=_chunk(0.1)))
    assert out == []


def test_max_segment_force_closes():
    # voz infinita → corta a max_segment_s
    seg = UtteranceSegmenter(
        vad_predict=lambda m: 0.9, sample_rate=SR, vad_col=2,
        speech_threshold=0.5, close_silence_ms=160, preroll_ms=0,
        max_segment_s=0.24, min_speech_ms=80,  # 3 chunks máx
    )
    out = []
    for i in range(6):
        out.extend(seg.feed(ts=1.0 + i * 0.08, chunk=_chunk(0.1)))
    assert len(out) >= 1
    assert out[0].audio.shape[0] <= CHUNK * 3


def test_during_tts_propagates():
    seg = UtteranceSegmenter(
        vad_predict=lambda m: 0.9, sample_rate=SR, vad_col=2,
        speech_threshold=0.5, close_silence_ms=80, preroll_ms=0,
        max_segment_s=30.0, min_speech_ms=80,
    )
    probs = iter([0.9, 0.9, 0.0])
    seg._vad_predict = lambda m: next(probs)
    out = []
    out.extend(seg.feed(ts=1.0, chunk=_chunk(0.1), tts_active=False))
    out.extend(seg.feed(ts=1.08, chunk=_chunk(0.1), tts_active=True))
    out.extend(seg.feed(ts=1.16, chunk=_chunk(0.0), tts_active=False))
    assert len(out) == 1
    assert out[0].during_tts is True


def test_vad_col_fallback_when_missing():
    # device 2ch (sin firmware 6ch): col 2 no existe → usa col 0 y avisa una vez
    seg = UtteranceSegmenter(
        vad_predict=lambda m: 0.0, sample_rate=SR, vad_col=2,
        speech_threshold=0.5, close_silence_ms=160, preroll_ms=0,
        max_segment_s=30.0, min_speech_ms=80,
    )
    two_ch = np.zeros((CHUNK, 2), dtype=np.float32)
    seg.feed(ts=1.0, chunk=two_ch)  # no debe lanzar
```

- [ ] **Step 2: Run test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_segmenter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ambient.segmenter'`

- [ ] **Step 3: Implementar `src/ambient/segmenter.py`**

```python
"""UtteranceSegmenter — Silero VAD chunked sobre mic crudo → RawSegment.

El XVF3800 post-DSP rompe a Silero (prob~0, visto 2026-06-04 en whisper_fast);
por eso el VAD del ambient path corre sobre la COLUMNA CRUDA (vad_col, default
2 = mic 0 del firmware 6ch). Con firmware 2ch la columna no existe y se degrada
a col 0 (peor señal para VAD, pero funcional).
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Callable

import numpy as np

from src.ambient.types import RawSegment

logger = logging.getLogger(__name__)


def make_silero_predictor() -> Callable[[np.ndarray], float]:
    """Factory del predictor real (Silero VAD vía torch.hub, patrón
    whisper_wake.py). Devuelve prob máxima entre las ventanas de 512 samples
    del chunk. Carga el modelo UNA vez al construirse.
    """
    import torch  # torch PRIMERO (regla orden imports CUDA del proyecto)

    model, _utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True,
    )
    model.eval()

    def predict(chunk_mono: np.ndarray) -> float:
        if chunk_mono.dtype != np.float32:
            chunk_mono = chunk_mono.astype(np.float32)
        probs = []
        for i in range(0, len(chunk_mono) - 511, 512):
            win = torch.from_numpy(chunk_mono[i : i + 512])
            with torch.no_grad():
                probs.append(float(model(win, 16000).item()))
        return max(probs) if probs else 0.0

    return predict


class UtteranceSegmenter:
    """Máquina de estados: silencio → voz → cola de silencio → RawSegment."""

    def __init__(
        self,
        vad_predict: Callable[[np.ndarray], float],
        sample_rate: int = 16000,
        vad_col: int = 2,
        speech_threshold: float = 0.5,
        close_silence_ms: int = 700,
        preroll_ms: int = 500,
        max_segment_s: float = 30.0,
        min_speech_ms: int = 300,
    ):
        self._vad_predict = vad_predict
        self.sample_rate = sample_rate
        self.vad_col = vad_col
        self.speech_threshold = speech_threshold
        self.close_silence_ms = close_silence_ms
        self.max_segment_s = max_segment_s
        self.min_speech_ms = min_speech_ms

        chunk_ms = 80.0  # CHUNK_SIZE=1280 @ 16kHz — mismo del pipeline
        self._preroll = deque(maxlen=max(0, int(round(preroll_ms / chunk_ms))))
        self._in_speech = False
        self._buf: list[tuple[float, np.ndarray, bool]] = []
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._col_warned = False

    def _vad_column(self, chunk: np.ndarray) -> np.ndarray:
        col = self.vad_col
        if chunk.ndim == 1:
            return chunk
        if chunk.shape[1] <= col:
            if not self._col_warned:
                self._col_warned = True
                logger.warning(
                    f"vad_col={col} no existe (device de {chunk.shape[1]}ch) "
                    f"— fallback a col 0 (sin mic crudo, VAD degradado)"
                )
            col = 0
        return chunk[:, col]

    def feed(
        self, ts: float, chunk: np.ndarray, tts_active: bool = False
    ) -> list[RawSegment]:
        """Procesar un chunk; devolver los segmentos cerrados (0, 1)."""
        chunk_ms = (chunk.shape[0] / self.sample_rate) * 1000
        prob = self._vad_predict(self._vad_column(chunk))
        is_speech = prob >= self.speech_threshold
        closed: list[RawSegment] = []

        if not self._in_speech:
            if is_speech:
                self._in_speech = True
                self._buf = list(self._preroll)
                self._buf.append((ts, chunk, tts_active))
                self._preroll.clear()
                self._silence_ms = 0.0
                self._speech_ms = chunk_ms
            else:
                if self._preroll.maxlen:
                    self._preroll.append((ts, chunk, tts_active))
            return closed

        # in_speech
        self._buf.append((ts, chunk, tts_active))
        if is_speech:
            self._silence_ms = 0.0
            self._speech_ms += chunk_ms
        else:
            self._silence_ms += chunk_ms

        dur_s = sum(c.shape[0] for _, c, _ in self._buf) / self.sample_rate
        if self._silence_ms >= self.close_silence_ms or dur_s >= self.max_segment_s:
            seg = self._close()
            if seg is not None:
                closed.append(seg)
        return closed

    def _close(self) -> RawSegment | None:
        buf, self._buf = self._buf, []
        self._in_speech = False
        self._silence_ms = 0.0
        speech_ms, self._speech_ms = self._speech_ms, 0.0
        if not buf or speech_ms < self.min_speech_ms:
            return None  # blip: menos voz que min_speech_ms → descartar
        t0 = buf[0][0]
        last_ts, last_chunk, _ = buf[-1]
        t1 = last_ts + last_chunk.shape[0] / self.sample_rate
        audio = np.concatenate([c for _, c, _ in buf], axis=0)
        during_tts = any(t for _, _, t in buf)
        return RawSegment(t0=t0, t1=t1, audio=audio, during_tts=during_tts)
```

- [ ] **Step 4: Run tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_segmenter.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ambient/segmenter.py tests/unit/ambient/test_segmenter.py
git commit -m "feat(ambient): UtteranceSegmenter — Silero VAD sobre mic crudo con pre-roll"
```

---

### Task 6: DoA por software (`src/ambient/doa.py`)

GCC-PHAT entre pares de mics crudos → azimut **relativo** + estabilidad (dispersión circular entre sub-ventanas). No requiere geometría absoluta del array: la firma direccional se calibra empíricamente (Fase 2) grabando con la TV sonando.

**Files:**
- Create: `src/ambient/doa.py`
- Test: `tests/unit/ambient/test_doa.py`

- [ ] **Step 1: Escribir el failing test**

```python
"""Tests: GCC-PHAT y DoAEstimator — azimut relativo desde mics crudos."""
import numpy as np

from src.ambient.doa import DoAEstimator, gcc_phat

SR = 16000


def _delayed_noise(n: int, delay_samples: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.3, size=n + abs(delay_samples) + 8).astype(np.float32)
    ref = base[: n]
    sig = base[delay_samples : delay_samples + n] if delay_samples >= 0 else base[: n]
    if delay_samples < 0:
        sig, ref = ref, base[-delay_samples : -delay_samples + n]
    return sig, ref


def test_gcc_phat_recovers_known_delay():
    delay = 4  # samples = 250µs @ 16k
    sig, ref = _delayed_noise(SR, delay)
    tau = gcc_phat(sig, ref, fs=SR, max_tau=0.001)
    assert abs(tau - (-delay / SR)) < 0.5 / SR  # sig adelantada → tau negativo


def test_estimator_stable_for_coherent_source():
    # Fuente coherente: mismo ruido con delays fijos entre canales → estabilidad alta
    rng = np.random.default_rng(11)
    n = SR * 2
    base = rng.normal(0, 0.3, size=n + 16).astype(np.float32)
    seg = np.zeros((n, 6), dtype=np.float32)
    # cols 2-5 = mics crudos; delays fijos simulan dirección estable
    seg[:, 2] = base[:n]
    seg[:, 3] = base[2 : n + 2]
    seg[:, 4] = base[4 : n + 4]
    seg[:, 5] = base[1 : n + 1]
    est = DoAEstimator(sample_rate=SR, raw_first_col=2, n_raw=4, win_s=0.5)
    res = est.estimate(seg)
    assert res is not None
    assert res.stability > 0.9
    assert -np.pi <= res.azimuth <= np.pi


def test_estimator_unstable_for_incoherent_noise():
    rng = np.random.default_rng(12)
    seg = rng.normal(0, 0.3, size=(SR * 2, 6)).astype(np.float32)
    est = DoAEstimator(sample_rate=SR, raw_first_col=2, n_raw=4, win_s=0.5)
    res = est.estimate(seg)
    assert res is not None
    assert res.stability < 0.9  # azimuts dispersos entre sub-ventanas


def test_estimator_returns_none_without_raw_channels():
    seg = np.zeros((SR, 2), dtype=np.float32)  # firmware 2ch: sin mics crudos
    est = DoAEstimator(sample_rate=SR, raw_first_col=2, n_raw=4)
    assert est.estimate(seg) is None
```

- [ ] **Step 2: Run test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_doa.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ambient.doa'`

- [ ] **Step 3: Implementar `src/ambient/doa.py`**

```python
"""DoA por software — GCC-PHAT sobre los mics crudos del XVF3800 (fw 6ch).

Azimut RELATIVO: atan2 de los TDOAs de los dos pares diagonales del array.
Sin geometría absoluta es consistente consigo mismo — suficiente para
clasificar contra una firma calibrada (tv_azimuth se mide en Fase 2 con la
TV sonando). stability = módulo del promedio circular entre sub-ventanas:
1.0 = dirección clavada (fuente puntual), ~0 = difuso/ruido.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gcc_phat(
    sig: np.ndarray,
    ref: np.ndarray,
    fs: int = 16000,
    max_tau: float | None = None,
    interp: int = 4,
) -> float:
    """TDOA (s) entre sig y ref por correlación cruzada generalizada PHAT."""
    n = sig.shape[0] + ref.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(ref, n=n)
    r = SIG * np.conj(REF)
    denom = np.abs(r)
    denom[denom < 1e-15] = 1e-15
    cc = np.fft.irfft(r / denom, n=interp * n)
    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc))) - max_shift
    return shift / float(interp * fs)


@dataclass
class DoAResult:
    azimuth: float    # rad relativo [-pi, pi]
    stability: float  # 0-1


class DoAEstimator:
    """Azimut relativo + estabilidad de un segmento multicanal."""

    def __init__(
        self,
        sample_rate: int = 16000,
        raw_first_col: int = 2,
        n_raw: int = 4,
        win_s: float = 0.5,
        mic_max_tau_s: float = 0.0005,  # ~17cm de apertura máx — clamp anti-outlier
    ):
        self.sample_rate = sample_rate
        self.raw_first_col = raw_first_col
        self.n_raw = n_raw
        self.win_s = win_s
        self.mic_max_tau_s = mic_max_tau_s

    def estimate(self, audio: np.ndarray) -> DoAResult | None:
        """None si el audio no trae los mics crudos (firmware 2ch)."""
        if audio.ndim != 2 or audio.shape[1] < self.raw_first_col + self.n_raw:
            return None
        c = self.raw_first_col
        m0, m1, m2, m3 = (audio[:, c + i] for i in range(4))

        win = int(self.win_s * self.sample_rate)
        if audio.shape[0] < win:
            win = audio.shape[0]
        azimuths = []
        for start in range(0, audio.shape[0] - win + 1, win):
            sl = slice(start, start + win)
            # pares diagonales del array: TDOA en ejes ~ortogonales
            tau02 = gcc_phat(m0[sl], m2[sl], fs=self.sample_rate, max_tau=self.mic_max_tau_s)
            tau13 = gcc_phat(m1[sl], m3[sl], fs=self.sample_rate, max_tau=self.mic_max_tau_s)
            azimuths.append(np.arctan2(tau13, tau02))
        if not azimuths:
            return None
        vec = np.exp(1j * np.array(azimuths))
        mean_vec = vec.mean()
        return DoAResult(
            azimuth=float(np.angle(mean_vec)),
            stability=float(np.abs(mean_vec)),
        )


def angular_distance(a: float, b: float) -> float:
    """Distancia angular |a-b| envuelta a [0, pi]."""
    d = abs(a - b) % (2 * np.pi)
    return float(min(d, 2 * np.pi - d))
```

- [ ] **Step 4: Run tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_doa.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ambient/doa.py tests/unit/ambient/test_doa.py
git commit -m "feat(ambient): DoA por software — GCC-PHAT con azimut relativo y estabilidad"
```

---

### Task 7: AmbientSTT (`src/ambient/ambient_stt.py`)

Wrapper async fino sobre `FastWhisperSTT` (instancia separada construida con `device=cuda:0`, `beam_size=1`, `initial_prompt=None`, `vad_filter=False` — el builder de Task 12 la arma). Extrae la columna ASR del segmento y transcribe en `asyncio.to_thread` (regla del proyecto: nunca bloquear el event loop).

**Files:**
- Create: `src/ambient/ambient_stt.py`
- Test: `tests/unit/ambient/test_ambient_stt.py`

- [ ] **Step 1: Escribir el failing test**

```python
"""Tests: AmbientSTT — wrapper async sobre FastWhisperSTT para el ambient path."""
import asyncio
from unittest.mock import MagicMock

import numpy as np

from src.ambient.ambient_stt import AmbientSTT
from src.stt.whisper_fast import STTResult


def _stt_mock(text: str = "hola mundo") -> MagicMock:
    m = MagicMock()
    m.transcribe_with_confidence.return_value = STTResult(
        text=text, elapsed_ms=42.0, no_speech_prob=0.1,
        avg_logprob=-0.3, compression_ratio=1.1,
    )
    return m


def test_transcribes_asr_column():
    inner = _stt_mock()
    astt = AmbientSTT(stt=inner, asr_col=1)
    audio = np.zeros((16000, 6), dtype=np.float32)
    audio[:, 1] = 0.5  # la columna ASR es reconocible

    result = asyncio.run(astt.transcribe(audio))

    assert result.text == "hola mundo"
    passed = inner.transcribe_with_confidence.call_args[0][0]
    assert passed.ndim == 1
    assert passed[0] == np.float32(0.5)


def test_falls_back_to_col0_when_asr_col_missing():
    inner = _stt_mock()
    astt = AmbientSTT(stt=inner, asr_col=1)
    mono_2d = np.full((16000, 1), 0.25, dtype=np.float32)

    asyncio.run(astt.transcribe(mono_2d))

    passed = inner.transcribe_with_confidence.call_args[0][0]
    assert passed[0] == np.float32(0.25)


def test_accepts_1d_audio():
    inner = _stt_mock()
    astt = AmbientSTT(stt=inner, asr_col=1)
    asyncio.run(astt.transcribe(np.zeros(16000, dtype=np.float32)))
    assert inner.transcribe_with_confidence.called
```

- [ ] **Step 2: Run test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_ambient_stt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ambient.ambient_stt'`

- [ ] **Step 3: Implementar `src/ambient/ambient_stt.py`**

```python
"""AmbientSTT — STT del ambient path (instancia separada en cuda:0).

NO comparte el FastWhisperSTT del command path (cuda:1): aislamiento por
contrato del spec — el ambient path jamás compite con la latencia <300ms.
El builder lo construye con beam_size=1 (velocidad), initial_prompt=None
(el prompt domótico causaba copias literales en audio ambiguo) y
vad_filter=False (el VAD ya lo hizo el segmenter sobre el mic crudo).
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np

from src.stt.whisper_fast import STTResult

logger = logging.getLogger(__name__)


class AmbientSTT:
    """Transcribe la columna ASR de un RawSegment, sin bloquear el loop."""

    def __init__(self, stt, asr_col: int = 1):
        """
        Args:
            stt: instancia FastWhisperSTT dedicada (cuda:0). DI por constructor.
            asr_col: columna del beam ASR en el audio multicanal (fw 6ch: 1).
        """
        self._stt = stt
        self.asr_col = asr_col

    def _asr_mono(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio
        col = self.asr_col if audio.shape[1] > self.asr_col else 0
        return np.ascontiguousarray(audio[:, col])

    async def transcribe(self, audio: np.ndarray) -> STTResult:
        """Transcripción con confianza, en thread (CTranslate2 es sync)."""
        mono = self._asr_mono(audio)
        return await asyncio.to_thread(self._stt.transcribe_with_confidence, mono)
```

- [ ] **Step 4: Run tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_ambient_stt.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ambient/ambient_stt.py tests/unit/ambient/test_ambient_stt.py
git commit -m "feat(ambient): AmbientSTT — instancia STT dedicada cuda:0, async"
```

---

### Task 8: SpeakerTagger (`src/ambient/speaker_tagger.py`)

Segunda instancia de `SpeakerIdentifier` (ECAPA, cuda:0). Sin voces enroladas (`data/users.json` no existe hoy) devuelve `("unknown", 0.0)` — el diseño lo contempla.

**Files:**
- Create: `src/ambient/speaker_tagger.py`
- Test: `tests/unit/ambient/test_speaker_tagger.py`

- [ ] **Step 1: Escribir el failing test**

```python
"""Tests: SpeakerTagger — etiqueta hablante por utterance con ECAPA."""
import asyncio
from unittest.mock import MagicMock

import numpy as np

from src.ambient.speaker_tagger import SpeakerTagger
from src.users.speaker_identifier import SpeakerMatch

SR = 16000


def _match(user_id, conf, known):
    return SpeakerMatch(
        user_id=user_id, confidence=conf,
        embedding=np.zeros(192, dtype=np.float32), is_known=known,
    )


def test_known_speaker():
    ident = MagicMock()
    ident.identify.return_value = _match("gabriel", 0.88, True)
    tagger = SpeakerTagger(
        identifier=ident,
        embeddings_loader=lambda: {"gabriel": np.ones(192, dtype=np.float32)},
        min_audio_s=0.8,
    )
    speaker, conf = asyncio.run(tagger.tag(np.zeros(SR * 2, dtype=np.float32)))
    assert speaker == "gabriel"
    assert conf == 0.88


def test_unknown_when_no_enrolled_embeddings():
    ident = MagicMock()
    tagger = SpeakerTagger(identifier=ident, embeddings_loader=lambda: {}, min_audio_s=0.8)
    speaker, conf = asyncio.run(tagger.tag(np.zeros(SR * 2, dtype=np.float32)))
    assert (speaker, conf) == ("unknown", 0.0)
    ident.identify.assert_not_called()  # sin enrolados no se gasta GPU


def test_unknown_for_too_short_audio():
    ident = MagicMock()
    tagger = SpeakerTagger(
        identifier=ident,
        embeddings_loader=lambda: {"gabriel": np.ones(192, dtype=np.float32)},
        min_audio_s=0.8,
    )
    speaker, conf = asyncio.run(tagger.tag(np.zeros(SR // 2, dtype=np.float32)))
    assert (speaker, conf) == ("unknown", 0.0)
    ident.identify.assert_not_called()


def test_identify_error_is_unknown_not_crash():
    ident = MagicMock()
    ident.identify.side_effect = RuntimeError("CUDA hiccup")
    tagger = SpeakerTagger(
        identifier=ident,
        embeddings_loader=lambda: {"gabriel": np.ones(192, dtype=np.float32)},
        min_audio_s=0.8,
    )
    speaker, conf = asyncio.run(tagger.tag(np.zeros(SR * 2, dtype=np.float32)))
    assert (speaker, conf) == ("unknown", 0.0)
```

- [ ] **Step 2: Run test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_speaker_tagger.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/ambient/speaker_tagger.py`**

```python
"""SpeakerTagger — etiqueta de hablante por utterance (ECAPA en cuda:0).

Instancia de SpeakerIdentifier SEPARADA de la del command path (aislamiento).
embeddings_loader es un callable (DI) que devuelve {user_id: embedding} — en
producción lee los enrolados; en tests, un dict fijo. Best-effort: cualquier
error → ("unknown", 0.0), nunca propaga.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerTagger:
    """(speaker, confidence) para el audio mono de una utterance."""

    def __init__(
        self,
        identifier,
        embeddings_loader: Callable[[], dict[str, np.ndarray]],
        min_audio_s: float = 0.8,
        sample_rate: int = 16000,
    ):
        self._identifier = identifier
        self._embeddings_loader = embeddings_loader
        self.min_audio_s = min_audio_s
        self.sample_rate = sample_rate

    async def tag(self, audio_mono: np.ndarray) -> tuple[str, float]:
        if len(audio_mono) < self.min_audio_s * self.sample_rate:
            return ("unknown", 0.0)
        try:
            registered = self._embeddings_loader() or {}
        except Exception as e:
            logger.debug(f"SpeakerTagger embeddings_loader error: {e}")
            return ("unknown", 0.0)
        if not registered:
            return ("unknown", 0.0)  # sin enrolados: no gastar GPU
        try:
            match = await asyncio.to_thread(
                self._identifier.identify, audio_mono, registered
            )
        except Exception as e:
            logger.debug(f"SpeakerTagger identify error (best-effort): {e}")
            return ("unknown", 0.0)
        if match.is_known and match.user_id:
            return (match.user_id, float(match.confidence))
        return ("unknown", float(match.confidence))
```

- [ ] **Step 4: Run tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_speaker_tagger.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ambient/speaker_tagger.py tests/unit/ambient/test_speaker_tagger.py
git commit -m "feat(ambient): SpeakerTagger — ECAPA por utterance, best-effort"
```

---

### Task 9: SourceClassifier (`src/ambient/source_classifier.py`)

Reglas del spec con la extensión `self`: `during_tts → self`; firma DoA ≈ TV (estable) y hablante no-conocido → `tv`; hablante conocido → `live`; DoA estable lejos de la TV → `live`; resto → `unknown`. Sin `tv_azimuth` calibrado (Fase 2), nunca clasifica `tv`.

**Files:**
- Create: `src/ambient/source_classifier.py`
- Test: `tests/unit/ambient/test_source_classifier.py`

- [ ] **Step 1: Escribir el failing test**

```python
"""Tests: SourceClassifier — matriz de reglas live|tv|self|unknown."""
import math

from src.ambient.source_classifier import SourceClassifier, SourceClassifierConfig


def _clf(tv_azimuth=1.0, **kw):
    cfg = SourceClassifierConfig(
        tv_azimuth=tv_azimuth,
        tv_tolerance_rad=kw.pop("tv_tolerance_rad", 0.35),
        min_stability=kw.pop("min_stability", 0.6),
        require_known_speaker_for_live=kw.pop("require_known_speaker_for_live", False),
    )
    return SourceClassifier(cfg)


def test_during_tts_is_self_regardless_of_everything():
    c = _clf()
    assert c.classify(speaker="gabriel", azimuth=1.0, stability=1.0, during_tts=True) == "self"


def test_stable_tv_direction_unknown_speaker_is_tv():
    c = _clf(tv_azimuth=1.0)
    assert c.classify(speaker="unknown", azimuth=1.1, stability=0.9, during_tts=False) == "tv"


def test_known_speaker_is_live_even_from_tv_direction():
    # El usuario sentado al lado de la TV no es la TV.
    c = _clf(tv_azimuth=1.0)
    assert c.classify(speaker="gabriel", azimuth=1.0, stability=0.9, during_tts=False) == "live"


def test_stable_non_tv_direction_is_live():
    c = _clf(tv_azimuth=1.0)
    assert c.classify(speaker="unknown", azimuth=-2.0, stability=0.9, during_tts=False) == "live"


def test_unstable_doa_is_unknown():
    c = _clf(tv_azimuth=1.0)
    assert c.classify(speaker="unknown", azimuth=1.0, stability=0.2, during_tts=False) == "unknown"


def test_no_doa_is_unknown():
    c = _clf(tv_azimuth=1.0)
    assert c.classify(speaker="unknown", azimuth=None, stability=0.0, during_tts=False) == "unknown"


def test_without_calibrated_tv_azimuth_never_tv():
    c = _clf(tv_azimuth=None)
    assert c.classify(speaker="unknown", azimuth=1.0, stability=0.95, during_tts=False) == "live"


def test_wraparound_angular_distance():
    # azimut pi y -pi son la misma dirección
    c = _clf(tv_azimuth=math.pi)
    assert c.classify(speaker="unknown", azimuth=-math.pi + 0.1, stability=0.9, during_tts=False) == "tv"


def test_require_known_speaker_for_live():
    c = _clf(tv_azimuth=1.0, require_known_speaker_for_live=True)
    assert c.classify(speaker="unknown", azimuth=-2.0, stability=0.9, during_tts=False) == "unknown"
    assert c.classify(speaker="gabriel", azimuth=-2.0, stability=0.9, during_tts=False) == "live"
```

- [ ] **Step 2: Run test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_source_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/ambient/source_classifier.py`**

```python
"""SourceClassifier — etiqueta la fuente de cada utterance (spec §4).

Orden de reglas (la primera que aplica gana):
  1. during_tts            → "self"   (nuestro TTS sonando; no va al RAG)
  2. speaker conocido      → "live"   (la persona manda sobre la dirección)
  3. sin DoA o inestable   → "unknown"
  4. DoA ≈ tv_azimuth      → "tv"     (requiere tv_azimuth calibrado en Fase 2)
  5. DoA estable no-TV     → "live" (o "unknown" si require_known_speaker_for_live)
"""
from __future__ import annotations

from dataclasses import dataclass

from src.ambient.doa import angular_distance


@dataclass
class SourceClassifierConfig:
    tv_azimuth: float | None = None        # rad relativo; None = sin calibrar
    tv_tolerance_rad: float = 0.35
    min_stability: float = 0.6
    require_known_speaker_for_live: bool = False


class SourceClassifier:
    """Reglas declarativas — sin estado, trivialmente testeable."""

    def __init__(self, config: SourceClassifierConfig):
        self._cfg = config

    def classify(
        self,
        *,
        speaker: str,
        azimuth: float | None,
        stability: float,
        during_tts: bool,
    ) -> str:
        cfg = self._cfg
        if during_tts:
            return "self"
        if speaker != "unknown":
            return "live"
        if azimuth is None or stability < cfg.min_stability:
            return "unknown"
        if (
            cfg.tv_azimuth is not None
            and angular_distance(azimuth, cfg.tv_azimuth) <= cfg.tv_tolerance_rad
        ):
            return "tv"
        return "unknown" if cfg.require_known_speaker_for_live else "live"
```

- [ ] **Step 4: Run tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_source_classifier.py -v`
Expected: 9 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ambient/source_classifier.py tests/unit/ambient/test_source_classifier.py
git commit -m "feat(ambient): SourceClassifier — reglas live|tv|self|unknown"
```

---

### Task 10: AmbientStore (`src/ambient/store.py`)

SQLite dedicada (`data/ambient.db`) vía aiosqlite (ya en requirements). TTL con purga explícita (`purge_expired()` — el transcriber la llama cada hora). Separada de `events.db` por ciclo de vida (decisión del spec §4).

**Files:**
- Create: `src/ambient/store.py`
- Test: `tests/unit/ambient/test_store.py`

- [ ] **Step 1: Escribir el failing test**

```python
"""Tests: AmbientStore — SQLite TTL para utterances ambientales."""
import asyncio
import time

from src.ambient.store import AmbientStore
from src.ambient.types import AmbientUtterance


def _utt(t0: float, text: str = "hola", source: str = "live", **kw) -> AmbientUtterance:
    return AmbientUtterance(
        room_id=kw.pop("room_id", "escritorio"), t0=t0, t1=t0 + 2.0,
        text=text, source=source, **kw,
    )


def _run(coro):
    return asyncio.run(coro)


def test_add_and_query_between(tmp_path):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "ambient.db"), retention_hours=12)
        await store.init()
        now = time.time()
        await store.add(_utt(now - 10, text="primera"))
        await store.add(_utt(now - 5, text="segunda"))
        await store.add(_utt(now - 5, text="otra room", room_id="living"))

        rows = await store.utterances_between("escritorio", now - 7, now)
        assert [r["text"] for r in rows] == ["segunda"]
        await store.close()
    _run(inner())


def test_undistilled_live_and_mark(tmp_path):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "a.db"), retention_hours=12)
        await store.init()
        now = time.time()
        id_live = await store.add(_utt(now, text="dato útil", source="live"))
        await store.add(_utt(now, text="ruido tele", source="tv"))
        await store.add(_utt(now, text="yo mismo", source="self"))

        batch = await store.undistilled_live(limit=10)
        assert [r["id"] for r in batch] == [id_live]

        await store.mark_distilled([id_live])
        assert await store.undistilled_live(limit=10) == []
        await store.close()
    _run(inner())


def test_purge_expired(tmp_path):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "a.db"), retention_hours=1)
        await store.init()
        now = time.time()
        await store.add(_utt(now - 7200, text="vieja"))   # 2h: expira
        await store.add(_utt(now - 60, text="fresca"))
        deleted = await store.purge_expired()
        assert deleted == 1
        rows = await store.utterances_between("escritorio", 0, now + 10)
        assert [r["text"] for r in rows] == ["fresca"]
        await store.close()
    _run(inner())


def test_add_validates_source(tmp_path):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "a.db"), retention_hours=1)
        await store.init()
        try:
            await store.add(_utt(time.time(), source="martian"))
            raised = False
        except ValueError:
            raised = True
        assert raised
        await store.close()
    _run(inner())
```

- [ ] **Step 2: Run test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/ambient/store.py`**

```python
"""AmbientStore — persistencia TTL de utterances (data/ambient.db).

Política del spec: destilar-y-descartar. El texto crudo vive acá
retention_hours y la purga lo borra; los hechos destilados viven en la
memoria de largo plazo (ChromaDB). Solo texto — jamás audio.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import aiosqlite

from src.ambient.types import SOURCE_VALUES, AmbientUtterance

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS utterances (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  room_id TEXT NOT NULL,
  t0 REAL NOT NULL,
  t1 REAL NOT NULL,
  text TEXT NOT NULL,
  speaker TEXT NOT NULL DEFAULT 'unknown',
  speaker_confidence REAL NOT NULL DEFAULT 0,
  azimuth REAL,
  azimuth_stability REAL NOT NULL DEFAULT 0,
  source TEXT NOT NULL DEFAULT 'unknown',
  confidence REAL,
  no_speech_prob REAL,
  during_tts INTEGER NOT NULL DEFAULT 0,
  distilled INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_utt_room_time ON utterances(room_id, t0);
CREATE INDEX IF NOT EXISTS idx_utt_distill ON utterances(distilled, source);
"""


class AmbientStore:
    """CRUD async sobre la tabla utterances, con TTL."""

    def __init__(self, db_path: str = "./data/ambient.db", retention_hours: float = 12.0):
        self.db_path = db_path
        self.retention_hours = retention_hours
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        logger.info(f"AmbientStore listo ({self.db_path}, TTL {self.retention_hours}h)")

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def add(self, utt: AmbientUtterance) -> int:
        if utt.source not in SOURCE_VALUES:
            raise ValueError(f"source inválido: {utt.source!r} (∉ {SOURCE_VALUES})")
        cur = await self._db.execute(
            """INSERT INTO utterances
               (room_id, t0, t1, text, speaker, speaker_confidence, azimuth,
                azimuth_stability, source, confidence, no_speech_prob,
                during_tts, distilled, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                utt.room_id, utt.t0, utt.t1, utt.text, utt.speaker,
                utt.speaker_confidence, utt.azimuth, utt.azimuth_stability,
                utt.source, utt.confidence, utt.no_speech_prob,
                int(utt.during_tts), int(utt.distilled), time.time(),
            ),
        )
        await self._db.commit()
        return cur.lastrowid

    async def utterances_between(
        self, room_id: str, t0: float, t1: float
    ) -> list[dict]:
        """Utterances de una room cuyo inicio cae en [t0, t1] (segunda opinión)."""
        cur = await self._db.execute(
            "SELECT * FROM utterances WHERE room_id=? AND t0>=? AND t0<=? ORDER BY t0",
            (room_id, t0, t1),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def undistilled_live(self, limit: int = 200) -> list[dict]:
        """Lote para el Distiller: solo source='live' sin destilar, viejas primero."""
        cur = await self._db.execute(
            "SELECT * FROM utterances WHERE distilled=0 AND source='live' "
            "ORDER BY t0 LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def mark_distilled(self, ids: list[int]) -> None:
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        await self._db.execute(
            f"UPDATE utterances SET distilled=1 WHERE id IN ({marks})", ids
        )
        await self._db.commit()

    async def purge_expired(self) -> int:
        """Borrar utterances más viejas que retention_hours. Devuelve filas."""
        cutoff = time.time() - self.retention_hours * 3600
        cur = await self._db.execute(
            "DELETE FROM utterances WHERE created_at < ?", (cutoff,)
        )
        await self._db.commit()
        if cur.rowcount:
            logger.info(f"AmbientStore purga: {cur.rowcount} utterances borradas (TTL)")
        return cur.rowcount
```

- [ ] **Step 4: Run tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_store.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ambient/store.py tests/unit/ambient/test_store.py
git commit -m "feat(ambient): AmbientStore — SQLite TTL destilar-y-descartar"
```

---

### Task 11: Distiller (`src/ambient/distiller.py`)

Job periódico: lotes `live` sin destilar → LLM **local** (:8101, decisión de privacidad del spec) con prompt de extracción → `store_fact` de `LongTermMemory` → `mark_distilled`. `chat_fn` inyectable (DI) para tests; la default usa aiohttp.

**Files:**
- Create: `src/ambient/distiller.py`
- Test: `tests/unit/ambient/test_distiller.py`

- [ ] **Step 1: Escribir el failing test**

```python
"""Tests: Distiller — extracción de hechos con LLM local y marcado."""
import asyncio
import json

from src.ambient.distiller import Distiller, _parse_facts


class FakeStore:
    def __init__(self, rows):
        self.rows = rows
        self.marked = []

    async def undistilled_live(self, limit=200):
        return self.rows[:limit]

    async def mark_distilled(self, ids):
        self.marked.extend(ids)


def _row(i, text):
    return {"id": i, "room_id": "escritorio", "t0": 1000.0 + i, "t1": 1002.0 + i,
            "text": text, "speaker": "gabriel", "source": "live"}


def test_distill_once_extracts_and_marks():
    rows = [_row(1, "che acordate que el viernes viene el plomero"),
            _row(2, "me encanta la pizza de acá")]
    store = FakeStore(rows)
    facts_out = []

    async def fake_chat(prompt: str) -> str:
        assert "plomero" in prompt
        return json.dumps([
            {"fact": "El viernes viene el plomero", "category": "fact", "confidence": 0.9},
            {"fact": "Le gusta la pizza del lugar habitual", "category": "preference", "confidence": 0.7},
        ])

    def fake_store_fact(fact, category, confidence=0.8, metadata=None):
        facts_out.append((fact, category, confidence, metadata))
        return f"fact_{len(facts_out)}"

    d = Distiller(store=store, chat_fn=fake_chat, store_fact_fn=fake_store_fact,
                  interval_hours=6, min_batch=1)
    n = asyncio.run(d.distill_once())
    assert n == 2
    assert store.marked == [1, 2]
    assert facts_out[0][1] == "fact"
    assert facts_out[1][1] == "preference"
    # metadata referencia el origen ambiental
    assert facts_out[0][3]["origin"] == "ambient"


def test_distill_below_min_batch_is_noop():
    store = FakeStore([_row(1, "hola")])

    async def fake_chat(prompt):
        raise AssertionError("no debe llamarse con batch < min_batch")

    d = Distiller(store=store, chat_fn=fake_chat, store_fact_fn=lambda *a, **k: "x",
                  interval_hours=6, min_batch=5)
    assert asyncio.run(d.distill_once()) == 0
    assert store.marked == []


def test_llm_error_does_not_mark():
    store = FakeStore([_row(1, "a"), _row(2, "b")])

    async def broken_chat(prompt):
        raise RuntimeError("LLM caído")

    d = Distiller(store=store, chat_fn=broken_chat, store_fact_fn=lambda *a, **k: "x",
                  interval_hours=6, min_batch=1)
    assert asyncio.run(d.distill_once()) == 0
    assert store.marked == []  # sin marcar: se reintenta el próximo ciclo


def test_parse_facts_tolerates_fences_and_garbage():
    raw = "```json\n[{\"fact\": \"X\", \"category\": \"fact\", \"confidence\": 0.8}]\n```"
    assert _parse_facts(raw) == [{"fact": "X", "category": "fact", "confidence": 0.8}]
    assert _parse_facts("no es json") == []
    assert _parse_facts("[]") == []
    # categorías inválidas se descartan, válidas quedan
    mixed = json.dumps([
        {"fact": "ok", "category": "preference", "confidence": 0.9},
        {"fact": "bad", "category": "alien", "confidence": 0.9},
        {"category": "fact", "confidence": 0.9},  # sin fact
    ])
    assert len(_parse_facts(mixed)) == 1
```

- [ ] **Step 2: Run test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_distiller.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/ambient/distiller.py`**

```python
"""Distiller — destilar-y-descartar (spec §1 privacidad, §4 componentes).

Cada interval_hours: toma utterances 'live' sin destilar, pide al LLM LOCAL
(:8101 — jamás cloud: conversaciones del hogar) hechos útiles en JSON, los
guarda en LongTermMemory (ChromaDB) y marca las filas. La purga del store
borra el crudo después. Si el LLM falla, NO marca — reintenta el próximo ciclo.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# Categorías válidas = las de MemoryFact (src/memory/memory_manager.py)
_VALID_CATEGORIES = {"personal", "preference", "pattern", "fact"}

_SYSTEM_PROMPT = (
    "Sos el extractor de memoria de un asistente del hogar. Recibís "
    "transcripciones ambientales (pueden tener errores de STT). Extraé SOLO "
    "hechos útiles y duraderos: preferencias, planes con fecha, datos de "
    "personas, patrones. NADA de charla trivial ni nada dudoso. Si no hay "
    "nada útil, devolvé []. Respondé SOLO un array JSON: "
    '[{"fact": str, "category": "personal|preference|pattern|fact", '
    '"confidence": 0.0-1.0}]'
)


def _parse_facts(raw: str) -> list[dict]:
    """Parse robusto del JSON del LLM (tolera fences y basura alrededor)."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    facts = []
    for item in data:
        if (
            isinstance(item, dict)
            and item.get("fact")
            and item.get("category") in _VALID_CATEGORIES
        ):
            facts.append({
                "fact": str(item["fact"]),
                "category": item["category"],
                "confidence": float(item.get("confidence", 0.7)),
            })
    return facts


def make_local_chat_fn(
    llm_url: str = "http://127.0.0.1:8101/v1",
    model: str = "local",
    timeout_s: float = 120.0,
) -> Callable[[str], Awaitable[str]]:
    """chat_fn real contra el llama-server local (OpenAI-compat)."""
    import aiohttp

    async def chat(prompt: str) -> str:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as session:
            async with session.post(
                f"{llm_url.rstrip('/')}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                },
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    return chat


class Distiller:
    """Job periódico de extracción de hechos."""

    def __init__(
        self,
        store,
        chat_fn: Callable[[str], Awaitable[str]],
        store_fact_fn: Callable[..., str],
        interval_hours: float = 6.0,
        min_batch: int = 5,
        max_batch_chars: int = 12000,
    ):
        """
        Args:
            store: AmbientStore (undistilled_live / mark_distilled).
            chat_fn: prompt → respuesta del LLM LOCAL. DI para tests.
            store_fact_fn: firma de LongTermMemory.store_fact(fact, category,
                confidence=, metadata=).
            min_batch: no destilar con menos utterances (ahorra ciclos LLM).
        """
        self._store = store
        self._chat = chat_fn
        self._store_fact = store_fact_fn
        self.interval_hours = interval_hours
        self.min_batch = min_batch
        self.max_batch_chars = max_batch_chars
        self._running = False

    async def distill_once(self) -> int:
        """Un ciclo de destilación. Devuelve cantidad de hechos guardados."""
        rows = await self._store.undistilled_live(limit=200)
        if len(rows) < self.min_batch:
            return 0
        # Truncar el lote por presupuesto de chars del prompt (7B local)
        batch, total = [], 0
        for r in rows:
            total += len(r["text"]) + 40
            if total > self.max_batch_chars:
                break
            batch.append(r)
        lines = [
            f"[{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['t0']))}] "
            f"({r['room_id']}, {r['speaker']}): {r['text']}"
            for r in batch
        ]
        prompt = "Transcripciones ambientales:\n" + "\n".join(lines)
        try:
            raw = await self._chat(prompt)
        except Exception as e:
            logger.warning(f"Distiller: LLM local falló ({e}) — reintento próximo ciclo")
            return 0
        facts = _parse_facts(raw)
        stored = 0
        for f in facts:
            try:
                self._store_fact(
                    f["fact"], f["category"], confidence=f["confidence"],
                    metadata={"origin": "ambient"},
                )
                stored += 1
            except Exception as e:
                logger.warning(f"Distiller: store_fact falló: {e}")
        # Marcar TODO el batch procesado (aunque no haya hechos: ya se evaluó)
        await self._store.mark_distilled([r["id"] for r in batch])
        if stored:
            logger.info(f"Distiller: {stored} hechos de {len(batch)} utterances")
        return stored

    async def run_forever(self) -> None:
        """Loop del job (lo lanza AmbientTranscriber.start)."""
        self._running = True
        while self._running:
            await asyncio.sleep(self.interval_hours * 3600)
            try:
                await self.distill_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Distiller: ciclo falló (best-effort, sigo)")

    def stop(self) -> None:
        self._running = False
```

- [ ] **Step 4: Run tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_distiller.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ambient/distiller.py tests/unit/ambient/test_distiller.py
git commit -m "feat(ambient): Distiller — extracción de hechos con LLM local, destilar-y-descartar"
```

---

### Task 12: AmbientTranscriber + builder (`src/ambient/transcriber.py`)

Orquestador: un worker async por room (drain tap → segmenter → STT+tagger+DoA → classify → store), purga horaria, job del distiller, y la señal en caliente `tv_active_recent()` para el shadow del wake. Best-effort por contrato: excepción → log + backoff, jamás propaga.

**Files:**
- Create: `src/ambient/transcriber.py`
- Test: `tests/unit/ambient/test_transcriber.py`

- [ ] **Step 1: Escribir el failing test**

```python
"""Tests: AmbientTranscriber — integración tap→segmenter→STT→store con fakes."""
import asyncio
import time

import numpy as np

from src.ambient.tap import MultiChannelTap
from src.ambient.segmenter import UtteranceSegmenter
from src.ambient.source_classifier import SourceClassifier, SourceClassifierConfig
from src.ambient.transcriber import AmbientTranscriber
from src.stt.whisper_fast import STTResult

SR = 16000
CHUNK = 1280


class FakeAmbientSTT:
    async def transcribe(self, audio):
        return STTResult(text="hola che", elapsed_ms=10.0,
                         no_speech_prob=0.05, avg_logprob=-0.2,
                         compression_ratio=1.0)


class FakeTagger:
    async def tag(self, mono):
        return ("unknown", 0.0)


class FakeDoA:
    def estimate(self, audio):
        from src.ambient.doa import DoAResult
        return DoAResult(azimuth=1.0, stability=0.95)


class FakeStore:
    def __init__(self):
        self.added = []

    async def add(self, utt):
        self.added.append(utt)
        return len(self.added)

    async def purge_expired(self):
        return 0


def _segmenter_factory():
    # VAD fake: voz si el chunk tiene energía
    def vad(mono):
        return 1.0 if float(np.abs(mono).max()) > 0.05 else 0.0
    return UtteranceSegmenter(
        vad_predict=vad, sample_rate=SR, vad_col=2, speech_threshold=0.5,
        close_silence_ms=160, preroll_ms=0, max_segment_s=30.0, min_speech_ms=80,
    )


def _make(store, tv_azimuth=2.5):
    tap = MultiChannelTap(maxlen_chunks=100)
    clf = SourceClassifier(SourceClassifierConfig(tv_azimuth=tv_azimuth))
    tr = AmbientTranscriber(
        tap=tap, segmenter_factory=_segmenter_factory,
        ambient_stt=FakeAmbientSTT(), tagger=FakeTagger(),
        doa_estimator=FakeDoA(), classifier=clf, store=store,
        rooms=["escritorio"], poll_interval_s=0.01,
    )
    return tap, tr


def test_voice_segment_lands_in_store_labeled():
    store = FakeStore()
    tap, tr = _make(store, tv_azimuth=2.5)  # DoA fake da 1.0 → no-TV → live

    async def inner():
        await tr.start()
        now = time.time()
        voz = np.full((CHUNK, 6), 0.2, dtype=np.float32)
        sil = np.zeros((CHUNK, 6), dtype=np.float32)
        for i, ch in enumerate([voz, voz, sil, sil, sil]):
            tap.push("escritorio", ch, ts=now + i * 0.08)
        # darle ciclos al worker
        for _ in range(50):
            await asyncio.sleep(0.02)
            if store.added:
                break
        await tr.stop()
    asyncio.run(inner())

    assert len(store.added) == 1
    u = store.added[0]
    assert u.text == "hola che"
    assert u.room_id == "escritorio"
    assert u.source == "live"
    assert u.azimuth == 1.0


def test_tv_direction_labels_tv_and_signal_fires():
    store = FakeStore()
    tap, tr = _make(store, tv_azimuth=1.0)  # DoA fake da 1.0 → TV

    async def inner():
        await tr.start()
        now = time.time()
        voz = np.full((CHUNK, 6), 0.2, dtype=np.float32)
        sil = np.zeros((CHUNK, 6), dtype=np.float32)
        for i, ch in enumerate([voz, voz, sil, sil, sil]):
            tap.push("escritorio", ch, ts=now + i * 0.08)
        for _ in range(50):
            await asyncio.sleep(0.02)
            if store.added:
                break
        # señal en caliente para el shadow del wake
        assert tr.tv_active_recent("escritorio", window_s=10.0) is True
        assert tr.tv_active_recent("living", window_s=10.0) is False
        await tr.stop()
    asyncio.run(inner())

    assert store.added[0].source == "tv"


def test_store_error_does_not_kill_worker():
    class BrokenStore(FakeStore):
        async def add(self, utt):
            raise RuntimeError("disco lleno")

    store = BrokenStore()
    tap, tr = _make(store)

    async def inner():
        await tr.start()
        now = time.time()
        voz = np.full((CHUNK, 6), 0.2, dtype=np.float32)
        sil = np.zeros((CHUNK, 6), dtype=np.float32)
        for i, ch in enumerate([voz, voz, sil, sil, sil]):
            tap.push("escritorio", ch, ts=now + i * 0.08)
        await asyncio.sleep(0.3)
        # el worker sigue vivo a pesar del error
        assert any(not t.done() for t in tr._tasks)
        await tr.stop()
    asyncio.run(inner())


def test_empty_text_is_not_stored():
    class EmptySTT:
        async def transcribe(self, audio):
            return STTResult(text="", elapsed_ms=5.0)

    store = FakeStore()
    tap = MultiChannelTap(maxlen_chunks=100)
    clf = SourceClassifier(SourceClassifierConfig())
    tr = AmbientTranscriber(
        tap=tap, segmenter_factory=_segmenter_factory,
        ambient_stt=EmptySTT(), tagger=FakeTagger(), doa_estimator=FakeDoA(),
        classifier=clf, store=store, rooms=["escritorio"], poll_interval_s=0.01,
    )

    async def inner():
        await tr.start()
        now = time.time()
        voz = np.full((CHUNK, 6), 0.2, dtype=np.float32)
        sil = np.zeros((CHUNK, 6), dtype=np.float32)
        for i, ch in enumerate([voz, voz, sil, sil, sil]):
            tap.push("escritorio", ch, ts=now + i * 0.08)
        await asyncio.sleep(0.3)
        await tr.stop()
    asyncio.run(inner())
    assert store.added == []
```

- [ ] **Step 2: Run test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_transcriber.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/ambient/transcriber.py`**

```python
"""AmbientTranscriber — orquestador del ambient path (spec §3-§5).

Un worker async por room: drena el tap, alimenta el segmenter y por cada
RawSegment corre STT (cuda:0) + speaker + DoA, clasifica la fuente y persiste.
Contrato best-effort: cualquier excepción → log + backoff exponencial; el
command path jamás se entera. build_ambient_path() arma el grafo completo
desde la config (DI por constructor en todos los niveles).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable

from src.ambient.types import AmbientUtterance

logger = logging.getLogger(__name__)

_PURGE_INTERVAL_S = 3600.0


class AmbientTranscriber:
    """Workers por room + purga + señal tv_active_recent()."""

    def __init__(
        self,
        tap,
        segmenter_factory: Callable[[], object],
        ambient_stt,
        tagger,
        doa_estimator,
        classifier,
        store,
        rooms: list[str],
        poll_interval_s: float = 0.25,
    ):
        self._tap = tap
        self._segmenter_factory = segmenter_factory
        self._stt = ambient_stt
        self._tagger = tagger
        self._doa = doa_estimator
        self._classifier = classifier
        self._store = store
        self._rooms = rooms
        self.poll_interval_s = poll_interval_s

        self._running = False
        self._tasks: list[asyncio.Task] = []
        # Última utterance 'tv' vista por room: (t1, ts_registro) — señal shadow
        self._last_tv: dict[str, float] = {}

    async def start(self) -> None:
        self._running = True
        for room_id in self._rooms:
            self._tap.register_room(room_id)
            self._tasks.append(asyncio.create_task(self._room_worker(room_id)))
        self._tasks.append(asyncio.create_task(self._purge_worker()))
        logger.info(f"AmbientTranscriber activo ({len(self._rooms)} rooms)")

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def tv_active_recent(self, room_id: str, window_s: float = 5.0) -> bool:
        """True si hubo una utterance 'tv' que terminó hace < window_s.
        Señal para el shadow anti-TV del wake (Fase 2: solo log)."""
        last = self._last_tv.get(room_id)
        return last is not None and (time.time() - last) < window_s

    async def _room_worker(self, room_id: str) -> None:
        segmenter = self._segmenter_factory()
        backoff = 1.0
        while self._running:
            try:
                items = self._tap.drain(room_id)
                segments = []
                for ts, chunk, tts_active in items:
                    segments.extend(
                        segmenter.feed(ts=ts, chunk=chunk, tts_active=tts_active)
                    )
                for seg in segments:
                    await self._handle_segment(room_id, seg)
                backoff = 1.0
                await asyncio.sleep(self.poll_interval_s)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception(
                    f"AmbientTranscriber[{room_id}]: error (best-effort, "
                    f"backoff {backoff:.0f}s)"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _handle_segment(self, room_id: str, seg) -> None:
        try:
            stt_result = await self._stt.transcribe(seg.audio)
            if not stt_result.text.strip():
                return
            mono = seg.audio[:, 0] if seg.audio.ndim == 2 else seg.audio
            speaker, sp_conf = await self._tagger.tag(mono)
            doa = await asyncio.to_thread(self._doa.estimate, seg.audio)
            azimuth = doa.azimuth if doa else None
            stability = doa.stability if doa else 0.0
            source = self._classifier.classify(
                speaker=speaker, azimuth=azimuth, stability=stability,
                during_tts=seg.during_tts,
            )
            utt = AmbientUtterance(
                room_id=room_id, t0=seg.t0, t1=seg.t1,
                text=stt_result.text.strip(),
                speaker=speaker, speaker_confidence=sp_conf,
                azimuth=azimuth, azimuth_stability=stability,
                source=source,
                confidence=stt_result.avg_logprob,
                no_speech_prob=stt_result.no_speech_prob,
                during_tts=seg.during_tts,
            )
            if source == "tv":
                self._last_tv[room_id] = time.time()
            await self._store.add(utt)
            logger.debug(
                f"[Ambient] {room_id} {source} {speaker}: "
                f"{utt.text[:60]!r} (az={azimuth}, stab={stability:.2f})"
            )
        except Exception:
            # un segmento malo no tira el worker — se pierde ese segmento
            logger.exception(f"AmbientTranscriber[{room_id}]: segmento descartado")

    async def _purge_worker(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(_PURGE_INTERVAL_S)
                await self._store.purge_expired()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("AmbientTranscriber: purga falló (sigo)")


@dataclass
class AmbientPath:
    """Grafo armado del ambient path — lo que main.py necesita."""
    tap: object
    transcriber: AmbientTranscriber
    store: object
    distiller: object | None

    async def start(self) -> None:
        await self.store.init()
        await self.transcriber.start()
        if self.distiller is not None:
            self.transcriber._tasks.append(
                asyncio.create_task(self.distiller.run_forever())
            )

    async def stop(self) -> None:
        if self.distiller is not None:
            self.distiller.stop()
        await self.transcriber.stop()
        await self.store.close()


def build_ambient_path(
    ambient_cfg: dict,
    stt_base_cfg: dict,
    room_ids: list[str],
    store_fact_fn: Callable[..., str] | None,
) -> AmbientPath:
    """Construir el ambient path completo desde config (wiring de main.py).

    Args:
        ambient_cfg: bloque `ambient:` de settings.yaml.
        stt_base_cfg: bloque `stt:` top-level (hereda model/compute_type/language).
        room_ids: rooms con mic real (las que tienen RoomStream).
        store_fact_fn: LongTermMemory.store_fact del MemoryManager de main, o
            None (sin memoria → sin distiller, solo buffer TTL).
    """
    from src.ambient.ambient_stt import AmbientSTT
    from src.ambient.distiller import Distiller, make_local_chat_fn
    from src.ambient.doa import DoAEstimator
    from src.ambient.segmenter import UtteranceSegmenter, make_silero_predictor
    from src.ambient.source_classifier import SourceClassifier, SourceClassifierConfig
    from src.ambient.speaker_tagger import SpeakerTagger
    from src.ambient.store import AmbientStore
    from src.ambient.tap import MultiChannelTap
    from src.stt.whisper_fast import FastWhisperSTT
    from src.users.speaker_identifier import SpeakerIdentifier

    stt_cfg = ambient_cfg.get("stt", {}) or {}
    # Hereda del stt top-level; overridea device/beam/prompt (desviación 3)
    ambient_whisper = FastWhisperSTT(
        model=stt_cfg.get("model", stt_base_cfg.get("model", "./models/whisper-v3-turbo")),
        device=stt_cfg.get("device", "cuda:0"),
        compute_type=stt_cfg.get("compute_type", stt_base_cfg.get("compute_type", "int8_float16")),
        language=stt_base_cfg.get("language", "es"),
        beam_size=stt_cfg.get("beam_size", 1),
        initial_prompt=None,
        vad_filter=False,
    )
    ambient_stt = AmbientSTT(stt=ambient_whisper, asr_col=ambient_cfg.get("asr_col", 1))

    seg_cfg = ambient_cfg.get("segmenter", {}) or {}
    vad_predict = make_silero_predictor()

    def segmenter_factory() -> UtteranceSegmenter:
        return UtteranceSegmenter(
            vad_predict=vad_predict,
            vad_col=ambient_cfg.get("vad_col", 2),
            speech_threshold=seg_cfg.get("speech_threshold", 0.5),
            close_silence_ms=seg_cfg.get("close_silence_ms", 700),
            preroll_ms=seg_cfg.get("preroll_ms", 500),
            max_segment_s=seg_cfg.get("max_segment_s", 30.0),
            min_speech_ms=seg_cfg.get("min_speech_ms", 300),
        )

    sp_cfg = ambient_cfg.get("speaker", {}) or {}
    identifier = SpeakerIdentifier(
        model_name="speechbrain/spkrec-ecapa-voxceleb",
        device=stt_cfg.get("device", "cuda:0"),
        similarity_threshold=0.75,
    )

    def embeddings_loader() -> dict:
        # Enrolamiento pendiente en el proyecto (data/users.json no existe);
        # loader vacío = speaker 'unknown' siempre, sin gastar GPU.
        import json
        from pathlib import Path

        import numpy as np

        emb_dir = Path(sp_cfg.get("embeddings_dir", "./data/users"))
        out = {}
        if emb_dir.is_dir():
            for f in emb_dir.glob("*_voice.npy"):
                out[f.stem.replace("_voice", "")] = np.load(f)
        return out

    tagger = SpeakerTagger(
        identifier=identifier,
        embeddings_loader=embeddings_loader,
        min_audio_s=sp_cfg.get("min_audio_s", 0.8),
    )

    clf_cfg = ambient_cfg.get("classifier", {}) or {}
    rooms_cfg = ambient_cfg.get("rooms", {}) or {}
    # tv_azimuth: por ahora una sola room con mic — usar la primera que lo tenga
    tv_azimuth = None
    for rid in room_ids:
        v = (rooms_cfg.get(rid, {}) or {}).get("tv_azimuth")
        if v is not None:
            tv_azimuth = float(v)
            break
    classifier = SourceClassifier(SourceClassifierConfig(
        tv_azimuth=tv_azimuth,
        tv_tolerance_rad=clf_cfg.get("tv_tolerance_rad", 0.35),
        min_stability=clf_cfg.get("min_stability", 0.6),
        require_known_speaker_for_live=clf_cfg.get("require_known_speaker_for_live", False),
    ))

    store = AmbientStore(
        db_path=ambient_cfg.get("db_path", "./data/ambient.db"),
        retention_hours=ambient_cfg.get("retention_hours", 12.0),
    )

    tap = MultiChannelTap()
    transcriber = AmbientTranscriber(
        tap=tap,
        segmenter_factory=segmenter_factory,
        ambient_stt=ambient_stt,
        tagger=tagger,
        doa_estimator=DoAEstimator(raw_first_col=ambient_cfg.get("raw_first_col", 2)),
        classifier=classifier,
        store=store,
        rooms=room_ids,
        poll_interval_s=ambient_cfg.get("poll_interval_s", 0.25),
    )

    distiller = None
    dis_cfg = ambient_cfg.get("distill", {}) or {}
    if store_fact_fn is not None:
        distiller = Distiller(
            store=store,
            chat_fn=make_local_chat_fn(
                llm_url=dis_cfg.get("llm_url", "http://127.0.0.1:8101/v1"),
                model=dis_cfg.get("model", "local"),
            ),
            store_fact_fn=store_fact_fn,
            interval_hours=dis_cfg.get("interval_hours", 6.0),
            min_batch=dis_cfg.get("min_batch", 5),
            max_batch_chars=dis_cfg.get("max_batch_chars", 12000),
        )

    return AmbientPath(tap=tap, transcriber=transcriber, store=store, distiller=distiller)
```

- [ ] **Step 4: Run tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_transcriber.py -v`
Expected: 4 PASS

- [ ] **Step 5: Run TODOS los tests del módulo**

Run: `.venv/bin/python -m pytest tests/unit/ambient/ -v`
Expected: todos PASS (≈33 tests)

- [ ] **Step 6: Commit**

```bash
git add src/ambient/transcriber.py tests/unit/ambient/test_transcriber.py
git commit -m "feat(ambient): AmbientTranscriber + build_ambient_path — orquestador best-effort"
```

---

### Task 13: Tee en MultiRoomAudioLoop + config + wiring main.py

Tres modificaciones quirúrgicas. El tee va **primero** en el audio callback (antes de barge-in/echo: el ambient path quiere TODO el audio) y propaga `tts_active` para la etiqueta `self`.

**Files:**
- Modify: `src/pipeline/multi_room_audio_loop.py` (método `attach_ambient` nuevo + tee en `_make_audio_callback`, hoy líneas 489-504)
- Modify: `config/settings.yaml` (sección `ambient:` nueva al final del archivo)
- Modify: `src/main.py` (wiring post-construcción de `multi_room_loop`, hoy línea ~918)
- Test: `tests/unit/ambient/test_loop_tee.py`

- [ ] **Step 1: Escribir el failing test**

```python
"""Tests: tee del audio callback de MultiRoomAudioLoop al MultiChannelTap."""
from unittest.mock import MagicMock

import numpy as np

from src.ambient.tap import MultiChannelTap
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream


def _loop_with_tap(tts_speaking: bool = False):
    detector = MagicMock()
    detector.detect.return_value = None
    echo = MagicMock()
    echo.is_safe_to_listen = True
    echo.should_process_audio.return_value = (True, "")
    rs = RoomStream(
        room_id="escritorio", device_index=4,
        wake_detector=detector, echo_suppressor=echo,
    )
    from src.conversation.follow_up_mode import FollowUpMode
    loop = MultiRoomAudioLoop(
        room_streams={"escritorio": rs},
        follow_up=FollowUpMode(follow_up_window=4.0),
    )
    rh = MagicMock()
    rh.is_speaking = tts_speaking
    loop.attach_response_handler(rh)
    tap = MultiChannelTap(maxlen_chunks=10)
    tap.register_room("escritorio")
    loop.attach_ambient(tap=tap, transcriber=None)
    return loop, rs, tap


def test_callback_tees_full_multichannel_chunk():
    loop, rs, tap = _loop_with_tap()
    cb = loop._make_audio_callback(rs)
    indata = np.random.default_rng(1).normal(0, 0.1, size=(1280, 6)).astype(np.float32)
    cb(indata, 1280, None, None)
    items = tap.drain("escritorio")
    assert len(items) == 1
    _, chunk, tts = items[0]
    assert chunk.shape == (1280, 6)  # multicanal completo, no solo capture_channel
    assert tts is False


def test_tee_marks_tts_active_and_still_tees_during_tts():
    # Durante TTS el flujo normal de wake hace return temprano — el tap
    # igual debe recibir el chunk, marcado tts_active=True.
    loop, rs, tap = _loop_with_tap(tts_speaking=True)
    loop.barge_in_enabled = True
    cb = loop._make_audio_callback(rs)
    indata = np.zeros((1280, 6), dtype=np.float32)
    cb(indata, 1280, None, None)
    items = tap.drain("escritorio")
    assert len(items) == 1
    assert items[0][2] is True


def test_no_tap_attached_keeps_callback_working():
    detector = MagicMock()
    detector.detect.return_value = None
    echo = MagicMock()
    echo.is_safe_to_listen = True
    echo.should_process_audio.return_value = (True, "")
    rs = RoomStream(room_id="escritorio", device_index=4,
                    wake_detector=detector, echo_suppressor=echo)
    from src.conversation.follow_up_mode import FollowUpMode
    loop = MultiRoomAudioLoop(
        room_streams={"escritorio": rs},
        follow_up=FollowUpMode(follow_up_window=4.0),
    )
    cb = loop._make_audio_callback(rs)
    cb(np.zeros((1280, 2), dtype=np.float32), 1280, None, None)  # no lanza
```

- [ ] **Step 2: Run test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_loop_tee.py -v`
Expected: FAIL — `AttributeError: 'MultiRoomAudioLoop' object has no attribute 'attach_ambient'`

- [ ] **Step 3: Modificar `src/pipeline/multi_room_audio_loop.py`**

3a. En `__init__`, después de `self._guard = ambient_guard` (línea ~202), agregar:

```python
        # Ambient path (spec 2026-06-06): tap multicanal + transcriber para la
        # señal shadow anti-TV. attach_ambient() post-init (orden de DI en main,
        # mismo patrón que attach_response_handler). None = feature apagada.
        self._ambient_tap = None
        self._ambient_transcriber = None
```

3b. Después del método `attach_response_handler` (línea ~220), agregar:

```python
    def attach_ambient(self, tap, transcriber=None) -> None:
        """Inyectar el ambient path post-init (tap obligatorio, transcriber
        opcional — habilita la señal shadow anti-TV en el wake)."""
        self._ambient_tap = tap
        self._ambient_transcriber = transcriber
```

3c. En `_make_audio_callback`, como PRIMERAS líneas del `audio_callback` (antes del bloque de `capture_channel`, línea ~493):

```python
        def audio_callback(indata, frames, time_info, status):
            # Tee al ambient path (spec 2026-06-06): SIEMPRE primero — el
            # ambient quiere todo el audio, incluso lo que el barge-in o el
            # echo suppressor descartan para el command path. O(1), fail-open.
            if self._ambient_tap is not None:
                tts_now = (
                    self._response_handler is not None
                    and self._response_handler.is_speaking
                )
                self._ambient_tap.push(
                    rs.room_id, indata.copy(), tts_active=tts_now
                )
```

(El resto del callback queda idéntico — el tee no altera el flujo del command path.)

- [ ] **Step 4: Run tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_loop_tee.py -v`
Expected: 3 PASS

- [ ] **Step 5: Agregar la sección `ambient:` a `config/settings.yaml`**

Al FINAL del archivo (`tail -5 config/settings.yaml` para confirmar el cierre actual), agregar:

```yaml
# ============================================================================
# Ambient path — transcripción continua multi-pista (spec 2026-06-06)
# Requiere firmware 6ch en el XVF3800 (runbook docs/runbooks/2026-06-06-*).
# Con firmware 2ch degrada solo: sin mics crudos → VAD sobre col 0, sin DoA.
# enabled: false = cero efecto (ni tap ni modelos cargados). Fase 2 lo
# prende con shadow_mode: true (etiqueta y persiste, no toca el command path).
# Fase 3 (flip, con datos de una semana): shadow_mode: false.
ambient:
  enabled: false
  shadow_mode: true
  asr_col: 1            # beam ASR (fw 6ch ch1 — mismo canal que el command path)
  vad_col: 2            # mic crudo 0 (Silero funciona sobre crudo, no post-DSP)
  raw_first_col: 2      # primera columna de mics crudos (DoA)
  retention_hours: 12   # TTL del texto crudo (destilar-y-descartar)
  poll_interval_s: 0.25
  db_path: "./data/ambient.db"
  stt:
    device: "cuda:0"    # GPU ambiental — JAMÁS cuda:1 (command path)
    beam_size: 1
    # model/compute_type heredan del bloque stt: top-level
  segmenter:
    speech_threshold: 0.5
    close_silence_ms: 700
    preroll_ms: 500
    max_segment_s: 30.0
    min_speech_ms: 300
  speaker:
    min_audio_s: 0.8
    embeddings_dir: "./data/users"
  classifier:
    tv_tolerance_rad: 0.35
    min_stability: 0.6
    require_known_speaker_for_live: false
  distill:
    llm_url: "http://127.0.0.1:8101/v1"   # SOLO local (privacidad del spec)
    model: "local"
    interval_hours: 6
    min_batch: 5
    max_batch_chars: 12000
  rooms:
    escritorio:
      tv_azimuth: null   # rad relativo — calibrar en Fase 2 (grabar con TV sonando)
```

- [ ] **Step 6: Wiring en `src/main.py`**

Ubicar el final del bloque que construye `multi_room_loop` (hoy: `logger.info(f"MultiRoomAudioLoop created...` línea ~915-918). Inmediatamente DESPUÉS de ese `logger.info`, dentro del mismo `if room_streams:`, agregar:

```python
            # Ambient path (spec 2026-06-06): transcripción continua multi-pista
            # en cuda:0. Best-effort: si falla el build, el command path sigue.
            ambient_path = None
            ambient_cfg = config.get("ambient", {}) or {}
            if ambient_cfg.get("enabled", False):
                try:
                    from src.ambient.transcriber import build_ambient_path
                    _store_fact = (
                        memory_manager.long_term.store_fact
                        if memory_manager is not None else None
                    )
                    ambient_path = build_ambient_path(
                        ambient_cfg=ambient_cfg,
                        stt_base_cfg=config.get("stt", {}) or {},
                        room_ids=list(room_streams.keys()),
                        store_fact_fn=_store_fact,
                    )
                    multi_room_loop.attach_ambient(
                        tap=ambient_path.tap,
                        transcriber=ambient_path.transcriber,
                    )
                    logger.info(
                        f"Ambient path construido ({len(room_streams)} rooms, "
                        f"shadow={ambient_cfg.get('shadow_mode', True)})"
                    )
                except Exception:
                    logger.exception(
                        "Ambient path no construido (best-effort) — "
                        "el pipeline de voz sigue sin él"
                    )
                    ambient_path = None
```

⚠️ Verificar el nombre real de la variable de memoria con `grep -n "memory_manager = " src/main.py` (hoy línea ~423: `memory_manager = MemoryManager(...)`). Si la construcción de `memory_manager` ocurre DESPUÉS de este punto en `main.py`, mover este bloque de wiring después de ella (el orden de DI manda) — o inicializar `_store_fact = None` y loguear que el distiller queda apagado.

Luego, ubicar dónde se arranca el loop: `grep -n "multi_room_loop.run\|multi_room_loop.start" src/main.py`. Junto al arranque, agregar:

```python
        if ambient_path is not None:
            asyncio.create_task(ambient_path.start())
```

Y en el shutdown (buscar `grep -n "multi_room_loop.stop\|finally" src/main.py` en la zona del run principal):

```python
        if ambient_path is not None:
            await ambient_path.stop()
```

Nota de scope: si `ambient_path` se define dentro del `if room_streams:` y el arranque/shutdown viven en otra función, declarar `ambient_path` donde sea visible en ambos puntos (mismo patrón que `multi_room_loop` ya resuelve — seguir ese precedente exacto).

- [ ] **Step 7: Verificar que main.py sigue importable y la suite ambient pasa**

Run: `.venv/bin/python -c "import ast; ast.parse(open('src/main.py').read()); print('main OK')" && .venv/bin/python -m pytest tests/unit/ambient/ -v`
Expected: `main OK` + todos PASS

- [ ] **Step 8: Commit**

```bash
git add src/pipeline/multi_room_audio_loop.py config/settings.yaml src/main.py tests/unit/ambient/test_loop_tee.py
git commit -m "feat(ambient): tee multicanal en el audio callback + config + wiring main"
```

---

### Task 14: Señal shadow anti-TV en el wake

Cuando `nexa.onnx` dispara con una utterance `tv` reciente, **loguear** lo que el guard habría hecho (shadow — igual que se hizo con `CommandGate`). El flip real a STRICT es Fase 3 (config, fuera de este plan).

**Files:**
- Modify: `src/pipeline/multi_room_audio_loop.py` (`_should_accept_wakeword`, hoy líneas 232-301)
- Test: `tests/unit/ambient/test_wake_shadow.py`

- [ ] **Step 1: Escribir el failing test**

```python
"""Tests: señal shadow anti-TV en _should_accept_wakeword (solo log, no bloquea)."""
import logging
import time
from unittest.mock import MagicMock

import numpy as np

from src.conversation.follow_up_mode import FollowUpMode
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream


def _loop():
    detector = MagicMock()
    echo = MagicMock()
    rs = RoomStream(room_id="escritorio", device_index=4,
                    wake_detector=detector, echo_suppressor=echo)
    return MultiRoomAudioLoop(
        room_streams={"escritorio": rs},
        follow_up=FollowUpMode(follow_up_window=4.0),
    )


def test_shadow_logs_but_accepts_when_tv_active(caplog):
    loop = _loop()
    transcriber = MagicMock()
    transcriber.tv_active_recent.return_value = True
    loop.attach_ambient(tap=MagicMock(), transcriber=transcriber)

    with caplog.at_level(logging.INFO):
        accepted = loop._should_accept_wakeword(
            "escritorio", rms=0.05, timestamp=time.time(), wake_score=0.55
        )
    assert accepted is True  # shadow: NUNCA bloquea
    assert any("[Ambient-shadow]" in r.message for r in caplog.records)


def test_no_log_when_tv_not_active(caplog):
    loop = _loop()
    transcriber = MagicMock()
    transcriber.tv_active_recent.return_value = False
    loop.attach_ambient(tap=MagicMock(), transcriber=transcriber)

    with caplog.at_level(logging.INFO):
        accepted = loop._should_accept_wakeword(
            "escritorio", rms=0.05, timestamp=time.time(), wake_score=0.55
        )
    assert accepted is True
    assert not any("[Ambient-shadow]" in r.message for r in caplog.records)


def test_transcriber_error_fails_open():
    loop = _loop()
    transcriber = MagicMock()
    transcriber.tv_active_recent.side_effect = RuntimeError("boom")
    loop.attach_ambient(tap=MagicMock(), transcriber=transcriber)
    accepted = loop._should_accept_wakeword(
        "escritorio", rms=0.05, timestamp=time.time(), wake_score=0.55
    )
    assert accepted is True  # fail-open, jamás afecta el wake
```

- [ ] **Step 2: Run test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_wake_shadow.py -v`
Expected: FAIL — no se loguea `[Ambient-shadow]` (el accept sí pasa: aún no existe el hook)

- [ ] **Step 3: Modificar `_should_accept_wakeword`**

En `src/pipeline/multi_room_audio_loop.py`, dentro de `_should_accept_wakeword`, DESPUÉS del bloque del AmbientGuard (línea ~264, tras el `return False` del guard) y ANTES del pre-gate `min_wake_rms`, agregar:

```python
        # Señal shadow anti-TV (spec 2026-06-06 §5.1, Fase 2): si el ambient
        # path vio una utterance 'tv' reciente, loguear qué haría el guard.
        # SOLO log — el flip a enforcement es Fase 3, con una semana de datos.
        # Fail-open: error del transcriber jamás toca el wake.
        if self._ambient_transcriber is not None:
            try:
                if self._ambient_transcriber.tv_active_recent(room_id):
                    logger.info(
                        f"[Ambient-shadow] wake en {room_id} con TV activa "
                        f"(score={wake_score:.2f}, rms={rms:.4f}) — "
                        f"enforcement habría exigido strict_wake_score"
                    )
            except Exception as e:
                logger.debug(f"[Ambient-shadow] señal no disponible: {e}")
```

- [ ] **Step 4: Run tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_wake_shadow.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/multi_room_audio_loop.py tests/unit/ambient/test_wake_shadow.py
git commit -m "feat(ambient): señal shadow anti-TV en el wake — log-only, fail-open"
```

---

### Task 15: Verificación final y push

- [ ] **Step 1: Suite ambient completa**

Run: `.venv/bin/python -m pytest tests/unit/ambient/ -v`
Expected: ≈39 PASS, 0 FAIL

- [ ] **Step 2: Tests de los módulos tocados (regresión)**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/ -v 2>/dev/null || .venv/bin/python -m pytest tests/ -k "multi_room or audio_loop" -v`
Expected: sin regresiones nuevas (la falla pre-existente `test_endpointing::test_voice_prob_uses_vad` del baseline no cuenta)

- [ ] **Step 3: Smoke del benchmark (laptop, sin GPU)**

Run: `.venv/bin/python tools/benchmark_ambient.py --doa --iterations 3`
Expected: imprime `DoA GCC-PHAT: ... ms por segmento de 3s (CPU)` sin error

- [ ] **Step 4: Push de la rama**

```bash
git push -u origin feat/ambient-multipista
```

---

## Qué queda FUERA de este plan (secuencia operacional posterior)

1. **Fase 0 en el server**: correr `tools/benchmark_ambient.py --stt --device cuda:0` con audio real del mic (`/tmp/nexa_debug/` tiene muestras). Si RTF > 0.5 → evaluar Parakeet v3 antes de seguir.
2. **Fase 1 (GATED)**: ejecutar el runbook de flasheo 6ch **con OK explícito del usuario**.
3. **Deploy + Fase 2**: `ambient.enabled: true` (shadow) en el server, una semana de datos.
4. **Calibración TV**: grabar con la TV sonando, calcular azimut mediano de las utterances → `ambient.rooms.escritorio.tv_azimuth`.
5. **Fase 3 (flip)**: conectar la señal `tv_active_recent` al AmbientGuard (STRICT situacional) en lugar del log shadow, activar enforcement con los datos de la semana. Eso es un mini-plan propio cuando haya números.
6. **Segunda opinión post-comando (spec §5.2)**: la API ya queda implementada en este plan (`AmbientStore.utterances_between`, Task 10) — conectarla al `RequestRouter` (comparar transcripción ambiental cuando el gate marca comando garbleado) es enforcement y va junto con Fase 3.
7. **Pre-gate VAD crudo para el wake (spec §5.3)**: la señal nace del mismo segmenter, pero "hay voz AHORA" requiere exponer estado en vivo (no segmentos cerrados). Se diseña junto con el enforcement de Fase 3 — con los datos de la semana shadow se decide si aporta sobre lo que ya da `tv_active_recent` + AmbientGuard.
8. **Enrolamiento de voces** (mejora el clasificador y habilita speaker en el RAG): pendiente preexistente del proyecto (20-30 frases, `scripts/enroll_voice`).



