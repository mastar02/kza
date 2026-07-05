# Robustez de transcripción del COMMAND PATH (Camino A + medición) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que el command path audio→texto casi nunca falle en silencio: recuperar capturas garble (temperature fallback de Whisper), reemplazar el descarte mudo por un earcon "te oí, no te entendí", y medir todo para decidir las fases siguientes con datos.

**Architecture:** 5 componentes independientes, cada uno detrás de un flag en `config/settings.yaml` (cero archivos de config nuevos): (1) temperature fallback en `whisper_fast.py`; (2) earcon gateado a "humano plausible" en el ResponseHandler + router; (3) voseo en `initial_prompt`; (4) endpointing más largo; (5) instrumentación de calidad ASR vía el `EventLogger` existente → `data/events.db`. Spec: `docs/superpowers/specs/2026-06-15-command-stt-robustness-design.md`.

**Tech Stack:** Python 3.13, faster-whisper (CTranslate2), numpy, pytest, SQLite (`data/events.db`), YAML config.

---

## File Structure

- `src/analytics/asr_quality.py` — **NUEVO**. Helper puro: traduce un outcome de captura (accepted / gate_rejected / low_confidence / earcon_fired / fallback_*) a un evento del `EventLogger`. Una sola responsabilidad: medición. Testeable con un `EventLogger` fake.
- `src/stt/whisper_fast.py` — temperature fallback (Comp. 1) + reportar si hubo fallback en `STTResult`.
- `src/pipeline/response_handler.py` — `play_earcon()` (Comp. 2).
- `src/tts/response_cache.py` — cargar el WAV del earcon en RAM al startup (Comp. 2).
- `src/pipeline/request_router.py` — gating del earcon + llamado a `asr_quality` en los puntos de descarte (Comp. 2 + 5).
- `tools/make_earcon.py` — **NUEVO**. Genera `data/earcons/not_understood.wav` (sin binarios en el repo; reproducible).
- `config/settings.yaml` — flags Comp. 1-5, voseo, `max_utterance_s`.
- Tests en `tests/unit/...` por componente.

**Orden de tareas (dependencias):** medición (Task 1) primero porque es el instrumento que valida el resto; luego fallback (Task 2); earcon asset+player (Task 3-4); gating+wiring (Task 5); voseo (Task 6); endpointing (Task 7).

---

## Task 1: Helper de medición de calidad ASR (`asr_quality.py`)

**Files:**
- Create: `src/analytics/asr_quality.py`
- Test: `tests/unit/analytics/test_asr_quality.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/analytics/test_asr_quality.py
from src.analytics.asr_quality import log_asr_outcome


class _FakeEventLogger:
    def __init__(self):
        self.calls = []

    def log(self, entity_id, action, event_type=None, trigger_phrase=None, extra_context=None):
        self.calls.append(
            dict(entity_id=entity_id, action=action, trigger_phrase=trigger_phrase,
                 extra_context=extra_context)
        )


def test_logs_one_event_with_room_and_reason():
    el = _FakeEventLogger()
    log_asr_outcome(el, room_id="living", outcome="gate_rejected",
                    reason="empty", text="", signals={"compression_ratio": None},
                    wake_score=0.81, rms=0.03)
    assert len(el.calls) == 1
    call = el.calls[0]
    assert call["entity_id"] == "asr_quality:living"
    assert call["action"] == "gate_rejected:empty"
    assert call["extra_context"]["wake_score"] == 0.81
    assert call["extra_context"]["rms"] == 0.03


def test_truncates_text_to_60_chars():
    el = _FakeEventLogger()
    log_asr_outcome(el, room_id="cocina", outcome="accepted", reason="ok",
                    text="x" * 200, signals={}, wake_score=1.0, rms=0.1)
    assert len(el.calls[0]["trigger_phrase"]) == 60


def test_none_event_logger_is_noop():
    # No debe explotar si no hay logger (fail-open): no exception.
    log_asr_outcome(None, room_id="hall", outcome="accepted", reason="ok",
                    text="dale", signals={}, wake_score=1.0, rms=0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/analytics/test_asr_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.analytics.asr_quality'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/analytics/asr_quality.py
"""Medición de calidad del command path STT.

Traduce el outcome de una captura post-wake a un evento del EventLogger
existente (data/events.db). NO toca el schema: usa entity_id="asr_quality:<room>"
y action="<outcome>:<reason>" para poder hacer luego:
    SELECT action, count(*) FROM events
    WHERE entity_id LIKE 'asr_quality:%' GROUP BY action;

Fail-open: si event_logger es None o falla, no se rompe el pipeline de voz.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_asr_outcome(
    event_logger,
    room_id: str,
    outcome: str,
    reason: str,
    text: str,
    signals: dict,
    wake_score: float,
    rms: float,
) -> None:
    """Registrar el resultado de una captura del command path.

    Args:
        event_logger: EventLogger o None (None = no-op, fail-open).
        room_id: room donde ocurrió la captura.
        outcome: accepted | gate_rejected | low_confidence | earcon_fired
                 | fallback_triggered | fallback_recovered.
        reason: sub-motivo (ej: empty, high_compression, ok).
        text: transcripción (se trunca a 60 chars).
        signals: dict de señales STT (compression_ratio, etc.).
        wake_score: score del wake que abrió la captura.
        rms: energía RMS de la captura.
    """
    if event_logger is None:
        return
    try:
        event_logger.log(
            entity_id=f"asr_quality:{room_id}",
            action=f"{outcome}:{reason}",
            trigger_phrase=(text or "")[:60],
            extra_context={
                "wake_score": wake_score,
                "rms": rms,
                "compression_ratio": signals.get("compression_ratio"),
                "no_speech_prob": signals.get("no_speech_prob"),
                "avg_logprob": signals.get("avg_logprob"),
            },
        )
    except Exception as e:  # nunca romper el pipeline de voz por la métrica
        logger.debug(f"[asr_quality] log falló (ignorado): {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/analytics/test_asr_quality.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/analytics/asr_quality.py tests/unit/analytics/test_asr_quality.py
git commit -m "feat(measure): helper de calidad ASR del command path → events.db"
```

---

## Task 2: Temperature fallback en Whisper (`whisper_fast.py`)

**Files:**
- Modify: `src/stt/whisper_fast.py` (constructor + `_transcribe_impl:166-202`, `STTResult:24-49`, `create_stt:471-490`)
- Test: `tests/unit/stt/test_whisper_fast_fallback.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/stt/test_whisper_fast_fallback.py
import numpy as np
from src.stt.whisper_fast import FastWhisperSTT, STTResult


class _FakeSegment:
    def __init__(self, text, no_speech_prob=1e-10, avg_logprob=-0.3, compression_ratio=1.1):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio


class _FakeModel:
    """Captura los kwargs de transcribe() para verificar el fallback."""
    def __init__(self):
        self.last_kwargs = None

    def transcribe(self, audio, **kwargs):
        self.last_kwargs = kwargs
        return iter([_FakeSegment("prendé la luz")]), object()


def _make(**over):
    stt = FastWhisperSTT(
        model="x", device="cpu",
        temperature_fallback=over.pop("temperature_fallback", True),
        fallback_temperatures=over.pop("fallback_temperatures", [0.0, 0.2, 0.4]),
        compression_ratio_threshold=over.pop("compression_ratio_threshold", 2.0),
        log_prob_threshold=over.pop("log_prob_threshold", -3.0),
    )
    stt._model = _FakeModel()
    return stt


def test_fallback_passes_temperature_list_and_compression_threshold():
    stt = _make()
    stt._transcribe_impl(np.zeros(1600, dtype=np.float32))
    kw = stt._model.last_kwargs
    assert kw["temperature"] == [0.0, 0.2, 0.4]
    assert kw["compression_ratio_threshold"] == 2.0
    assert kw["log_prob_threshold"] == -3.0


def test_fallback_disabled_uses_scalar_zero():
    stt = _make(temperature_fallback=False)
    stt._transcribe_impl(np.zeros(1600, dtype=np.float32))
    kw = stt._model.last_kwargs
    assert kw["temperature"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/stt/test_whisper_fast_fallback.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'temperature_fallback'`

- [ ] **Step 3: Write minimal implementation**

En `src/stt/whisper_fast.py`, ampliar el constructor (`__init__`, líneas 54-64) agregando los params **al final** (defaults seguros = comportamiento actual si se deja `temperature_fallback=False`, pero el factory lo pondrá en True):

```python
    def __init__(
        self,
        model: str = "distil-whisper/distil-small.en",
        device: str = "cuda:0",
        compute_type: str = "float16",
        language: str = "es",
        beam_size: int = 1,
        best_of: int = 1,
        initial_prompt: str | None = None,
        vad_filter: bool = True,
        temperature_fallback: bool = False,
        fallback_temperatures: list[float] | None = None,
        compression_ratio_threshold: float = 2.0,
        log_prob_threshold: float = -3.0,
    ):
```

Guardar los nuevos atributos junto a los demás (después de `self.vad_filter = vad_filter`, línea 86):

```python
        self.temperature_fallback = temperature_fallback
        self.fallback_temperatures = fallback_temperatures or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.compression_ratio_threshold = compression_ratio_threshold
        self.log_prob_threshold = log_prob_threshold
```

En `_transcribe_impl` (líneas 166-180), reemplazar el bloque de `self._model.transcribe(...)` por:

```python
        # Temperature: lista (fallback adaptativo de Whisper) o escalar 0.
        # En turbo, no_speech/avg_logprob están muertos → el fallback se gatea
        # por compression_ratio (única señal viva). log_prob_threshold se deja
        # muy bajo para NO disparar fallback espurio por el logprob invertido.
        temperature = (
            self.fallback_temperatures if self.temperature_fallback else 0
        )
        segments, info = self._model.transcribe(
            audio_input,
            language=self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=temperature,
            compression_ratio_threshold=self.compression_ratio_threshold,
            log_prob_threshold=self.log_prob_threshold,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
            vad_filter=self.vad_filter,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 100,
                "threshold": 0.5,
            },
        )
```

En `create_stt` (líneas 481-490), pasar los nuevos params desde config:

```python
        return FastWhisperSTT(
            model=config.get("model", "distil-whisper/distil-small.en"),
            device=config.get("device", "cuda:0"),
            compute_type=config.get("compute_type", "float16"),
            language=config.get("language", "es"),
            beam_size=config.get("beam_size", 1),
            best_of=config.get("best_of", 1),
            initial_prompt=config.get("initial_prompt"),
            vad_filter=config.get("vad_filter", True),
            temperature_fallback=config.get("temperature_fallback", {}).get("enabled", False),
            fallback_temperatures=config.get("temperature_fallback", {}).get("temperatures"),
            compression_ratio_threshold=config.get("temperature_fallback", {}).get("compression_ratio_threshold", 2.0),
            log_prob_threshold=config.get("temperature_fallback", {}).get("log_prob_threshold", -3.0),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/stt/test_whisper_fast_fallback.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the existing STT suite (no regresión)**

Run: `pytest tests/ -k "whisper" -v`
Expected: PASS (incluye los tests previos de whisper_fast)

- [ ] **Step 6: Add config flag**

En `config/settings.yaml`, dentro del bloque `stt:` (después de `vad_filter: false`, ~línea 161), agregar:

```yaml
  # Temperature fallback (2026-06-15): re-decode adaptativo de Whisper sobre
  # segmentos garble. Gateado SOLO por compression_ratio (única señal viva en
  # turbo); log_prob_threshold muy bajo para no disparar por el logprob
  # invertido. Caso fácil = temp 0 (0 latencia extra); garble = sube temp.
  temperature_fallback:
    enabled: true
    temperatures: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    compression_ratio_threshold: 2.0
    log_prob_threshold: -3.0
```

- [ ] **Step 7: Commit**

```bash
git add src/stt/whisper_fast.py tests/unit/stt/test_whisper_fast_fallback.py config/settings.yaml
git commit -m "feat(stt): temperature fallback gateado por compression_ratio (recupera garble far-field)"
```

---

## Task 3: Generador del WAV del earcon (`tools/make_earcon.py`)

**Files:**
- Create: `tools/make_earcon.py`
- Output: `data/earcons/not_understood.wav`

- [ ] **Step 1: Write the generator (no test — es un tool reproducible)**

```python
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
```

- [ ] **Step 2: Generate the asset**

Run: `python -m tools.make_earcon`
Expected: `earcon escrito: data/earcons/not_understood.wav (200ms)`

- [ ] **Step 3: Commit (incluye el WAV — chico, ~10KB, y el tool)**

```bash
git add tools/make_earcon.py data/earcons/not_understood.wav
git commit -m "feat(earcon): generador + asset del earcon 'no entendí' (200ms)"
```

---

## Task 4: `ResponseHandler.play_earcon()` + carga en cache

**Files:**
- Modify: `src/tts/response_cache.py` (cargar el WAV al startup)
- Modify: `src/pipeline/response_handler.py` (método `play_earcon`)
- Test: `tests/unit/pipeline/test_play_earcon.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/pipeline/test_play_earcon.py
import numpy as np
from src.pipeline.response_handler import ResponseHandler


def _make_handler():
    # ResponseHandler con un earcon pre-cargado y un zone_manager fake que
    # registra qué se reprodujo.
    h = ResponseHandler.__new__(ResponseHandler)
    h._earcon_audio = np.zeros(4800, dtype=np.float32)  # 200ms @ 24k
    h._earcon_sr = 24000
    h._active_zone_id = "zone_living"
    h.zone_manager = None
    h.played = []

    def _fake_play_array(audio, sr, zone_id):
        h.played.append((len(audio), sr, zone_id))

    h._play_earcon_array = _fake_play_array
    return h


def test_play_earcon_plays_loaded_asset():
    h = _make_handler()
    ResponseHandler.play_earcon(h, zone_id="zone_cocina")
    assert h.played == [(4800, 24000, "zone_cocina")]


def test_play_earcon_noop_when_no_asset():
    h = _make_handler()
    h._earcon_audio = None
    ResponseHandler.play_earcon(h, zone_id="zone_cocina")
    assert h.played == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/pipeline/test_play_earcon.py -v`
Expected: FAIL — `AttributeError: type object 'ResponseHandler' has no attribute 'play_earcon'`

- [ ] **Step 3: Implement `play_earcon` + asset loading**

En `src/tts/response_cache.py`, agregar una función módulo-nivel para cargar el WAV (al lado de la clase `ResponseCache`):

```python
import wave as _wave


def load_earcon(path: str) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Cargar un WAV de earcon a float32 mono. Fail-open: (None, None) si falla."""
    try:
        with _wave.open(path, "rb") as w:
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, sr
    except Exception as e:
        logger.warning(f"[earcon] no se pudo cargar {path!r}: {e}")
        return None, None
```

En `src/pipeline/response_handler.py`:
1. En `__init__` (constructor, ~línea 35-85), agregar params y carga:

```python
        # Earcon "no entendí" (2026-06-15): se carga una vez a RAM.
        self._earcon_audio = None
        self._earcon_sr = 24000
        earcon_path = (config or {}).get("earcon_asset") if hasattr(self, "_noop") else None
        # NOTA: el path real se pasa desde main.py vía set_earcon(); ver Task 5.
```

   (Para no depender del orden de DI, exponer un setter explícito.)

2. Agregar el setter y el método de playback:

```python
    def set_earcon(self, audio, sr: int) -> None:
        """Inyectar el earcon pre-cargado (np.ndarray float32, sample rate)."""
        self._earcon_audio = audio
        self._earcon_sr = sr

    def _play_earcon_array(self, audio, sr: int, zone_id: str) -> None:
        """Reproducir el array del earcon en la zona (bypassa TTS y hooks)."""
        # Reusa el path de playback de audio crudo del handler. _playback_raw
        # existe para acks cacheados; si no, cae al _speak_direct de PCM.
        self._playback_pcm(audio, sr, zone_id)

    def play_earcon(self, zone_id: str = None, room_context=None) -> None:
        """Reproducir el earcon 'no entendí'. No-op si no hay asset."""
        if self._earcon_audio is None:
            return
        if room_context and hasattr(room_context, "room_id") and not zone_id:
            zone_id = f"zone_{room_context.room_id}"
        target_zone = zone_id or self._active_zone_id
        self._is_speaking = True
        try:
            self._play_earcon_array(self._earcon_audio, self._earcon_sr, target_zone)
        finally:
            self._is_speaking = False
```

3. Implementar `_playback_pcm` reusando el camino de `_playback_cached`/`_speak_to_zone`. Si ya existe `_playback_cached(cached, zone)` que toma un `CachedAudio` (ver `response_handler.py:243`), envolver el array:

```python
    def _playback_pcm(self, audio, sr: int, zone_id: str) -> None:
        from src.tts.response_cache import CachedAudio
        cached = CachedAudio(audio=audio, sample_rate=sr, duration_s=len(audio) / sr)
        self._playback_cached(cached, zone_id)
```

   (Verificar el dataclass `CachedAudio` en `response_cache.py` — campos `audio`, `sample_rate`, `duration_s`. Ajustar nombres si difieren.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/pipeline/test_play_earcon.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/tts/response_cache.py src/pipeline/response_handler.py tests/unit/pipeline/test_play_earcon.py
git commit -m "feat(earcon): ResponseHandler.play_earcon() + carga del asset a RAM"
```

---

## Task 5: Gating "humano plausible" + wiring en el router (earcon + medición)

**Files:**
- Create: `src/pipeline/earcon_gate.py` (lógica pura de decisión)
- Modify: `src/pipeline/request_router.py` (call sites `:449-455`, `:563`, `:588`)
- Modify: `src/main.py` (cargar earcon + pasar al ResponseHandler + flags)
- Test: `tests/unit/pipeline/test_earcon_gate.py`

- [ ] **Step 1: Write the failing test (lógica de gating pura)**

```python
# tests/unit/pipeline/test_earcon_gate.py
from src.pipeline.earcon_gate import should_play_earcon


CFG = dict(enabled=True, min_wake_score=0.55, min_rms=0.02,
           reasons=("empty", "empty_after_norm", "high_compression", "low_confidence"))


def test_fires_on_empty_with_strong_wake_and_energy():
    assert should_play_earcon("empty", wake_score=0.81, rms=0.05, cfg=CFG) is True


def test_silent_on_noise_phrase_even_with_strong_wake():
    # TV/eco: NUNCA earcon.
    assert should_play_earcon("noise_phrase:'gracias por ver'", wake_score=0.9,
                              rms=0.1, cfg=CFG) is False


def test_silent_on_filler():
    assert should_play_earcon("filler_word:'si'", wake_score=0.9, rms=0.1, cfg=CFG) is False


def test_silent_on_weak_wake():
    assert should_play_earcon("empty", wake_score=0.41, rms=0.05, cfg=CFG) is False


def test_silent_on_low_energy():
    assert should_play_earcon("empty", wake_score=0.81, rms=0.005, cfg=CFG) is False


def test_high_compression_prefix_matches():
    assert should_play_earcon("high_compression:3.40>2.2", wake_score=0.81,
                              rms=0.05, cfg=CFG) is True


def test_disabled_never_fires():
    cfg = {**CFG, "enabled": False}
    assert should_play_earcon("empty", wake_score=0.99, rms=0.5, cfg=cfg) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/pipeline/test_earcon_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline.earcon_gate'`

- [ ] **Step 3: Implement the gate**

```python
# src/pipeline/earcon_gate.py
"""Decide si suena el earcon 'no entendí'.

Regla "humano plausible": el earcon NUNCA puede sonarle a la TV. Suena solo si
el wake fue fuerte Y hubo energía real Y el motivo del reject es de
no-comprensión (no de ruido/eco). Lógica pura, un solo lugar testeable.
"""
from __future__ import annotations

# Motivos que indican TV/eco/ruido → JAMÁS earcon (aunque el wake sea fuerte).
_NOISE_PREFIXES = ("noise_phrase", "filler_word", "word_repetition", "missing_wake")


def should_play_earcon(reason: str, wake_score: float, rms: float, cfg: dict) -> bool:
    """True si corresponde reproducir el earcon para este reject.

    Args:
        reason: AcceptanceDecision.reason o el intent de reject del router
            (ej: 'empty', "high_compression:3.4>2.2", 'low_confidence:0.42').
        wake_score: score del wake que abrió la captura.
        rms: energía RMS de la captura.
        cfg: {enabled, min_wake_score, min_rms, reasons}.
    """
    if not cfg.get("enabled", False):
        return False
    if any(reason.startswith(p) for p in _NOISE_PREFIXES):
        return False
    if wake_score < cfg.get("min_wake_score", 0.55):
        return False
    if rms < cfg.get("min_rms", 0.02):
        return False
    allowed = cfg.get("reasons", ())
    return any(reason.startswith(r) for r in allowed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/pipeline/test_earcon_gate.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Wire into the router (gate reject + low confidence)**

En `src/pipeline/request_router.py`:
1. Constructor (`__init__`, ~línea 219): agregar params `earcon_cfg: dict | None = None`, `asr_event_logger=None` y guardarlos:

```python
        self._earcon_cfg = earcon_cfg or {"enabled": False}
        self._asr_event_logger = asr_event_logger
```

2. En el reject del CommandGate (`:449-455`), antes de `return result`, insertar la decisión de earcon + medición. El `wake_score`/`rms` deben llegar al `route(...)`; si hoy no están en la firma, agregarlos como kwargs opcionales (`wake_score: float = 1.0`, `rms: float = 0.0`) y propagarlos desde `multi_room_audio_loop` (ver Step 6). Reemplazar el bloque:

```python
        if not gate_decision.accept:
            result["intent"] = "gate_rejected"
            result["response"] = ""
            from src.pipeline.earcon_gate import should_play_earcon
            from src.analytics.asr_quality import log_asr_outcome
            _reason = gate_decision.reason
            if should_play_earcon(_reason, wake_score, rms, self._earcon_cfg):
                _rc = None
                if self.room_context_manager and room_id:
                    _rc = self.room_context_manager.resolve_room(mic_zone_id=room_id, user_id=None)
                self.response_handler.play_earcon(room_context=_rc)
                log_asr_outcome(self._asr_event_logger, room_id, "earcon_fired",
                                _reason, text, gate_decision.signals, wake_score, rms)
            log_asr_outcome(self._asr_event_logger, room_id, "gate_rejected",
                            _reason, text, gate_decision.signals, wake_score, rms)
            return result
```

   Repetir el mismo patrón en el segundo call site del gate (`:760-766`, path legacy).

3. En el reject por confianza baja (`:556-565`), reemplazar `result["response"] = ""` por el earcon (motivo `low_confidence`):

```python
            if not grammar_backed and classification.confidence < self.llm_min_command_confidence:
                _reason = f"low_confidence:{classification.confidence:.2f}"
                logger.info(f"[LLMRouter] rechazado por confianza baja: {text!r}")
                result["intent"] = _reason
                result["success"] = False
                result["response"] = ""
                from src.pipeline.earcon_gate import should_play_earcon
                from src.analytics.asr_quality import log_asr_outcome
                if should_play_earcon(_reason, wake_score, rms, self._earcon_cfg):
                    _rc = None
                    if self.room_context_manager and room_id:
                        _rc = self.room_context_manager.resolve_room(mic_zone_id=room_id, user_id=None)
                    self.response_handler.play_earcon(room_context=_rc)
                    log_asr_outcome(self._asr_event_logger, room_id, "earcon_fired",
                                    _reason, text, {}, wake_score, rms)
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result
```

4. En `:588` (intent sin verbo): la decisión "solo sonido" → reemplazar la voz por earcon (configurable). Cambiar:

```python
                # 2026-06-15: decisión usuario "solo sonido" → earcon en vez de voz.
                result["response"] = ""
                _rc = None
                if self.room_context_manager and room_id:
                    _rc = self.room_context_manager.resolve_room(mic_zone_id=room_id, user_id=None)
                self.response_handler.play_earcon(room_context=_rc)
```

- [ ] **Step 6: Propagar `wake_score`/`rms` al router + cargar earcon en main.py**

En `src/main.py`:
1. Tras construir `response_handler`, cargar el earcon y setearlo:

```python
    from src.tts.response_cache import load_earcon
    _earcon_cfg = config.get("command_gate", {}).get("earcon", {"enabled": False})
    _earcon_audio, _earcon_sr = load_earcon(_earcon_cfg.get("asset", "data/earcons/not_understood.wav"))
    if _earcon_audio is not None:
        response_handler.set_earcon(_earcon_audio, _earcon_sr)
```

2. Pasar al `RequestRouter(...)` (línea 1219-1250) los nuevos kwargs:

```python
        earcon_cfg=_earcon_cfg,
        asr_event_logger=event_logger,
```

3. En `src/pipeline/multi_room_audio_loop.py`, donde se invoca `request_router.route(...)` tras la captura, propagar `wake_score=` y `rms=` (ya disponibles en `_should_accept_wakeword` y el ciclo de captura). Si la firma de `route` no los acepta aún, agregarlos como kwargs opcionales (Step 5.2).

- [ ] **Step 7: Run the router + earcon suites**

Run: `pytest tests/unit/pipeline/test_earcon_gate.py tests/unit/pipeline/test_play_earcon.py -v && pytest tests/ -k "router and gate" -v`
Expected: PASS (sin regresión en los tests del router/gate existentes)

- [ ] **Step 8: Add earcon config**

En `config/settings.yaml`, dentro del bloque `command_gate:` (después de `max_compression_ratio: 2.2`, ~línea 275):

```yaml
  # Earcon "te oí, no te entendí" (2026-06-15). Reemplaza el descarte mudo por
  # un sonido corto. Gateado a "humano plausible": NUNCA suena ante TV/eco
  # (noise_phrase/filler/word_repetition) ni capturas de baja energía.
  # min_wake_score 0.55 = wake fuerte (base 0.40); min_rms 0.02 apenas bajo el
  # piso inflado por el AGC (~0.025-0.05). CALIBRAR ambos con repro.
  earcon:
    enabled: true
    asset: "data/earcons/not_understood.wav"
    min_wake_score: 0.55
    min_rms: 0.02
    reasons: ["empty", "empty_after_norm", "high_compression", "low_confidence"]
```

- [ ] **Step 9: Commit**

```bash
git add src/pipeline/earcon_gate.py src/pipeline/request_router.py src/main.py src/pipeline/multi_room_audio_loop.py config/settings.yaml tests/unit/pipeline/test_earcon_gate.py
git commit -m "feat(earcon): gating humano-plausible + wiring en router + medición (mata el 'nada' mudo)"
```

---

## Task 6: Voseo en `initial_prompt` (A/B, solo vocabulario)

**Files:**
- Modify: `config/settings.yaml` (`stt.initial_prompt`, líneas 168-171)
- Test: `tests/unit/stt/test_initial_prompt_voseo.py` (guard de contenido)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/stt/test_initial_prompt_voseo.py
import yaml
from pathlib import Path


def _prompt() -> str:
    cfg = yaml.safe_load(Path("config/settings.yaml").read_text())
    return cfg["stt"]["initial_prompt"].lower()


def test_prompt_has_voseo_vocabulary():
    p = _prompt()
    for tok in ("prendé", "apagá", "subí", "bajá", "poné"):
        assert tok in p, f"falta voseo: {tok}"


def test_prompt_has_no_verbatim_command_phrases():
    # Guard anti-fantasma (incidente 2026-05-29): nunca frases-comando verbatim.
    p = _prompt()
    for banned in ("nexa prendé", "nexa subí", "nexa apagá"):
        assert banned not in p, f"frase-comando verbatim prohibida: {banned}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/stt/test_initial_prompt_voseo.py -v`
Expected: FAIL en `test_prompt_has_voseo_vocabulary` (el prompt actual no tiene voseo)

- [ ] **Step 3: Edit the prompt (solo vocabulario, sin frases-comando)**

En `config/settings.yaml`, reemplazar el `initial_prompt` (líneas 168-171) por:

```yaml
  initial_prompt: >-
    Esto es un asistente de voz llamado Nexa que controla luces, aire
    acondicionado, persianas y música en el escritorio, el living, la cocina,
    el baño y el hall. Habla rioplatense con voseo: prendé, apagá, subí, bajá,
    poné, dale.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/stt/test_initial_prompt_voseo.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add config/settings.yaml tests/unit/stt/test_initial_prompt_voseo.py
git commit -m "feat(stt): voseo rioplatense en initial_prompt (solo vocabulario, A/B-able)"
```

---

## Task 7: Endpointing más largo (`max_utterance_s`)

**Files:**
- Modify: `config/settings.yaml` (`rooms.wake_word` / bloque de captura, `max_utterance_s`)

- [ ] **Step 1: Localizar y editar el valor**

Run: `grep -n "max_utterance_s" config/settings.yaml`
Expected: una línea `max_utterance_s: 3.5`

Editarla a:

```yaml
    # 2026-06-15: 3.5 → 5.0. Comandos largos/con pausa natural se truncaban a
    # 3.5s → STT veía audio parcial → vacío/equivocado → descarte mudo.
    # Revertir si sube gate_rejected:noise (más charla ambiente captada).
    max_utterance_s: 5.0
```

- [ ] **Step 2: Validate config loads (schema Pydantic)**

Run: `python -c "from src.config_loader import load_config; load_config('config/settings.yaml'); print('OK')"`
Expected: `OK` (o el comando de validación de schema del proyecto; ver `f53b373 feat(core): validate settings.yaml against Pydantic schema at boot`)

- [ ] **Step 3: Commit**

```bash
git add config/settings.yaml
git commit -m "feat(capture): max_utterance_s 3.5→5.0 (no truncar comandos largos)"
```

---

## Self-Review (hecho)

**1. Spec coverage:**
- Comp. 1 (temperature fallback) → Task 2 ✓
- Comp. 2 (earcon + gating humano-plausible) → Tasks 3, 4, 5 ✓
- Comp. 3 (voseo) → Task 6 ✓
- Comp. 4 (endpointing) → Task 7 ✓
- Comp. 5 (medición) → Task 1 + wiring en Task 5 ✓
- Criterios de salto a B/C → quedan en el spec §4; se evalúan leyendo `events.db` tras un finde (no es tarea de código).

**2. Placeholder scan:** sin TBD/TODO. Las únicas notas "verificar/ajustar" son sobre nombres internos a confirmar contra el código real (`CachedAudio` fields, firma de `route`), explícitas y con instrucción concreta, no placeholders de lógica.

**3. Type consistency:** `log_asr_outcome(event_logger, room_id, outcome, reason, text, signals, wake_score, rms)` se usa idéntico en Task 1 y Task 5. `should_play_earcon(reason, wake_score, rms, cfg)` idéntico en Task 5. `play_earcon(zone_id=None, room_context=None)` y `set_earcon(audio, sr)` consistentes entre Task 4 y Task 5. Config keys (`temperature_fallback.*`, `command_gate.earcon.*`) consistentes entre tasks y spec.

**Riesgo de integración conocido (anotado para el ejecutor):** Tasks 4 y 5 tocan internals de `response_handler.py`/`multi_room_audio_loop.py` cuyas firmas exactas (`_playback_cached`, `CachedAudio`, la invocación de `route()`) deben confirmarse al abrir cada archivo; los nombres usados son los observados pero pueden requerir ajuste menor. Cada task tiene su test que falla primero, así que un desajuste se detecta de inmediato.
