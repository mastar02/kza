# S1 — Model warmup at startup

**Effort**: 🟢 30 min
**Depende**: nada
**Branch sugerido**: `feat/s1-model-warmup`

## Objetivo

Eliminar el lag de ~600 ms del primer comando post-restart. CUDA compila kernels
la primera vez que se usa un modelo → la primera transcripción de Whisper es
~3× más lenta que las siguientes. Pre-ejecutar cada modelo con input dummy al
boot resuelve esto.

## Arquitectura

Después de cada `.load()` de modelo en `src/main.py`, correr una inferencia con
audio de silencio (1 s a 16 kHz = 16000 samples de ceros). Logear el tiempo.

Modelos a warmpear:
1. **STT** (FastWhisperSTT) — `stt.transcribe(silence_1s)` para cache de Whisper.
2. **TTS** (Kokoro) — `tts.synthesize("hola")` para cachear kernels.
3. **SpeakerID** (ECAPA) — `speaker_identifier.get_embedding(silence_1s)` si se
   creó (puede ser None si speaker_id deshabilitado).
4. **Emotion** (wav2vec2) — `emotion_detector.detect(silence_1s, 16000)`.
5. **BGE-M3** (embeddings) — `chroma.embedder.encode(["warmup"])`.

El orden importa: STT y TTS son los críticos del path caliente (cada comando los
usa). SpeakerID + Emotion son secundarios. BGE-M3 se usa solo para queries a
Chroma.

## Tareas

### 1. Agregar función `_warmup_models()` en `src/main.py`

Ubicación: justo después de todos los `.load()` y antes del `VoicePipeline.run()`.

```python
async def _warmup_models(stt, tts, speaker_identifier, emotion_detector, chroma):
    """
    Precalentar modelos con input dummy para evitar cold-start lag en el
    primer comando real. CUDA compila kernels la primera vez — este warmup
    hace que todas las compilaciones ocurran al startup, no en runtime.
    """
    import numpy as np
    silence = np.zeros(16000, dtype=np.float32)  # 1s @ 16kHz

    timings = {}
    t0 = time.perf_counter()
    try:
        stt.transcribe(silence, sample_rate=16000)
        timings["stt"] = (time.perf_counter() - t0) * 1000
    except Exception as e:
        logger.warning(f"Warmup STT skipped: {e}")

    t0 = time.perf_counter()
    try:
        # TTS: generar 1 palabra (no reproducir). Puede ser sync o async.
        maybe = tts.synthesize("hola") if hasattr(tts, "synthesize") else None
        if asyncio.iscoroutine(maybe):
            await maybe
        timings["tts"] = (time.perf_counter() - t0) * 1000
    except Exception as e:
        logger.warning(f"Warmup TTS skipped: {e}")

    if speaker_identifier:
        t0 = time.perf_counter()
        try:
            speaker_identifier.get_embedding(silence)
            timings["speaker_id"] = (time.perf_counter() - t0) * 1000
        except Exception as e:
            logger.warning(f"Warmup speaker_id skipped: {e}")

    if emotion_detector:
        t0 = time.perf_counter()
        try:
            if hasattr(emotion_detector, "detect"):
                emotion_detector.detect(silence, 16000)
            timings["emotion"] = (time.perf_counter() - t0) * 1000
        except Exception as e:
            logger.warning(f"Warmup emotion skipped: {e}")

    if chroma and hasattr(chroma, "_embedder") and chroma._embedder is not None:
        t0 = time.perf_counter()
        try:
            chroma._embedder.encode(["warmup"])
            timings["bge_m3"] = (time.perf_counter() - t0) * 1000
        except Exception as e:
            logger.warning(f"Warmup BGE-M3 skipped: {e}")

    summary = " ".join(f"{k}={v:.0f}ms" for k, v in timings.items())
    logger.info(f"Warmup: {summary}")
```

### 2. Invocar antes de `pipeline.run()`

```python
# Existing code:
# ...all .load() calls...
# pipeline = VoicePipeline(...)

# NEW:
warmup_config = config.get("warmup", {})
if warmup_config.get("enabled", True):
    await _warmup_models(stt, tts, speaker_identifier, emotion_detector, chroma)

# Start presence detector before pipeline
if presence_detector:
    await presence_detector.start()
```

### 3. Config flag

`config/settings.yaml`:
```yaml
# Warmup de modelos al startup. Elimina cold-start de ~600ms en el primer
# comando post-restart. Desactivar solo para debug.
warmup:
  enabled: true
```

## Validación

### Antes
1. `systemctl --user restart kza-voice.service`
2. Esperar que inicialice.
3. Decir "nexa apagá la luz" inmediatamente → latencia típica 600-800ms STT.

### Después
1. Mismo restart.
2. Log debe mostrar: `Warmup: stt=350ms tts=80ms speaker_id=40ms emotion=95ms bge_m3=65ms` (una sola vez).
3. Decir comando → STT ~180-220ms (similar al comando número 10).

### Test unitario

`tests/unit/test_warmup.py` (opcional — la validación es más integración):
```python
import numpy as np
from unittest.mock import MagicMock
from src.main import _warmup_models  # export si está inner

def test_warmup_calls_each_model():
    stt = MagicMock()
    stt.transcribe = MagicMock(return_value=("", 0.0))
    tts = MagicMock()
    # ... etc
    import asyncio
    asyncio.run(_warmup_models(stt, tts, None, None, None))
    stt.transcribe.assert_called_once()
```

## Edge cases

- **STT con VAD filter activo**: con silencio puro, faster-whisper puede devolver
  segmentos vacíos. OK, igual compila kernels.
- **TTS que requiere speaker embedding**: si hay, pasar placeholder o skip.
- **BGE-M3 con texto vacío**: `encode([""])` puede fallar; usar `["warmup"]`.
- **Orden de imports**: `time`, `asyncio`, `numpy` ya están importados en main.py.
- **Excepciones**: cada warmup en try/except individual — un modelo que falle no
  debe romper el startup.

## Commit message sugerido

```
feat(startup): warmup de modelos post-load para eliminar cold start

Primer comando post-restart tardaba ~600ms por compilación JIT de kernels
CUDA. Pre-ejecutamos cada modelo con silencio dummy al startup para que
toda la compilación quede del lado del boot.

- src/main.py: _warmup_models() corre STT, TTS, SpeakerID, Emotion, BGE-M3
  con input dummy. Log unificado "Warmup: stt=Xms tts=Yms...".
- config/settings.yaml: warmup.enabled (true por default).

Latencia del primer comando cae de ~600ms a ~200ms — equivalente al
régimen estable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Checklist

- [ ] Implementar `_warmup_models()` en `src/main.py`
- [ ] Invocar antes de `pipeline.run()`
- [ ] Agregar `warmup.enabled` en `config/settings.yaml`
- [ ] Verificar que todos los modelos tienen el método esperado (STT.transcribe, TTS.synthesize, etc.)
- [ ] Correr tests existentes (`pytest tests/unit/nlu/ tests/unit/wakeword/`)
- [ ] Commit + push
- [ ] Post-merge: deploy + verificar log `Warmup: ...` + latencia primer comando
