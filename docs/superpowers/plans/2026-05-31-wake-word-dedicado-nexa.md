# Wake-word dedicado "Nexa" (openWakeWord) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **⚠️ Entorno:** se ejecuta **en vivo en el server** (`kza@192.168.1.2`, `~/app`), Python 3.13 venv `.venv/bin/python`. Repo local drifteado — comparar `git hash-object` antes de tocar. VRAM en cuda:1 apretadísima — preflight antes de reiniciar `kza-voice.service`.

**Goal:** Reemplazar "Whisper-como-wake-word" (que alucina ~500/h sobre silencio) por un modelo wake-word dedicado "Nexa" (openWakeWord, ONNX, CPU, siempre-on) que no se dispara sobre silencio, eliminando el flood de alucinaciones en el origen y bajando la latencia de wake de ~580ms a <150ms.

**Architecture:** La arquitectura de wake-dedicado **ya está cableada** en el código (`main.py:668` selector por `engine`; `WakeWordDetector` con `.detect()→(word,score)`; `multi_room_audio_loop.py` graba el comando en estado `listening` con endpointing VAD `_check_vad_completion` y lo transcribe con `FastWhisperSTT` que usa `vad_filter=True`). El path Whisper-específico (texto pretranscripto) usa `getattr(...,None)` → openwakeword cae con gracia al STT normal sobre el clip grabado. **El único componente faltante es el modelo "Nexa".onnx.** Este plan: (1) valida el path openwakeword con un modelo placeholder, (2) entrena "Nexa", (3) deploya y calibra.

**Tech Stack:** Python 3.13, openwakeword 0.6 (ONNX, NO tflite — sin wheels 3.13), onnxruntime 1.24.3 (ya instalado y validado funcionando), faster-whisper (STT post-wake, ya con vad_filter=True), Kokoro TTS (síntesis de positivos), Piper español, systemd `--user` (`kza-voice.service`), métricas JSONL (`~/logs/kza-metrics.jsonl`).

---

## Estado verificado (recon 2026-05-31)

- ✅ openWakeWord funciona en Python 3.13 con ONNX: `Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")` + `predict(silencio)=0.0` (NO se dispara sobre silencio).
- ✅ `WakeWordDetector` (`src/wakeword/detector.py`): `load()`, `predict()→dict`, `detect()→tuple[str,float]|None` (línea 214), `reset()`. Escala float32→int16 internamente.
- ✅ Modelos base presentes en venv: `hey_jarvis_v0.1.onnx`, `melspectrogram.onnx`, `embedding_model.onnx`, `silero_vad.onnx`.
- ✅ `main.py` selector: `engine=="whisper"` → WhisperWakeDetector; `else` → `WakeWordDetector(models=[model], threshold=...)`.
- ✅ `settings.yaml:1002-1011`: escritorio `engine: "whisper"`, `model: "hey_jarvis"`, `threshold: 0.35` (openwakeword params ya presentes).
- ✅ `multi_room_audio_loop.py`: estado `listening` graba comando + `_check_vad_completion` (endpointing) + `_dispatch_command`; STT vía `self._stt` cuando no hay pretranscribed_text.
- ❌ `models/wakeword/` vacío — no hay modelo "Nexa".

---

## File Structure

| Archivo | Responsabilidad | Tasks |
|---------|-----------------|-------|
| `config/settings.yaml` | `wake_word.engine`, `model`, `threshold` del escritorio | 0, 4 |
| `models/wakeword/nexa.onnx` | Modelo wake-word custom (output del training) | 3 |
| `scripts/train_wake_word.py` (nuevo) | Wrapper de entrenamiento openWakeWord (síntesis + train + export ONNX) | 1, 3 |
| `scripts/eval_wake_word.py` (nuevo) | Evaluar recall/FA del modelo sobre held-out + silencio real del server | 3 |
| `data/wake_training/nexa_real/` (nuevo) | Grabaciones reales de "Nexa" (dependencia usuario) | 2 |
| `src/pipeline/multi_room_audio_loop.py` | (Solo si Task 5 detecta endpointing flojo) | 5 |

---

## Fase 0 — Validar el path openwakeword end-to-end con placeholder (1 deploy, reversible)

> Probar que TODA la cadena openwakeword funciona (detección → grabación → STT → NLU → HA) **antes** de invertir en entrenar "Nexa", usando el modelo `hey_jarvis` que ya existe. Reversible en 1 línea de config. Requiere que un humano diga "hey jarvis" para el smoke test.

### Task 0.1: Flip temporal a openwakeword + hey_jarvis

**Files:**
- Modify: `config/settings.yaml` (bloque `wake_word` del escritorio, ~línea 1003)

- [ ] **Step 1: Backup + cambiar engine**

Run (server):
```bash
ssh kza 'cd ~/app && cp config/settings.yaml config/settings.yaml.bak.$(date +%s) && \
  sed -i "s/    engine: \"whisper\".*/    engine: \"openwakeword\"   # F0 validación temporal (revertir a whisper si falla)/" config/settings.yaml && \
  grep -n "engine:" config/settings.yaml | head'
```
Expected: la línea del wake_word del escritorio ahora dice `engine: "openwakeword"`. (Verificar que NO se cambió el `engine: "dual"` del TTS ni otros — el sed apunta a `"whisper"`.)

- [ ] **Step 2: Preflight VRAM + restart**

Run:
```bash
ssh kza 'nvidia-smi --query-gpu=memory.free --format=csv,noheader -i 1 && \
  systemctl --user restart kza-voice.service && sleep 10 && \
  systemctl --user is-active kza-voice.service'
```
Expected: `active`. Si <800MB libres en cuda:1, ver `project_stt_double_load_oom` antes de reiniciar.

- [ ] **Step 3: Confirmar que el detector openwakeword cargó (no Whisper)**

Run:
```bash
ssh kza 'journalctl --user -u kza-voice.service --since "1 min ago" --no-pager | grep -iE "OpenWakeWord cargado|WakeWordDetector|wake word detector" | head'
```
Expected: log `OpenWakeWord cargado: ['hey_jarvis']`. NO debe aparecer `WhisperWakeDetector`.

- [ ] **Step 4: Smoke test (humano) — "hey jarvis"**

Decir en el escritorio: **"Hey Jarvis, prendé la luz del escritorio"**. Verificar:
```bash
ssh kza 'journalctl --user -u kza-voice.service -f' | grep -iE "detect|listening|GRAMMAR_FASTPATH|HA-CALL|wake"
```
Expected: detección del wake (`hey_jarvis score=...`), `listening=True`, comando grabado, STT transcribe "prendé la luz del escritorio", acción HA. **La luz se prende.**

- [ ] **Step 5: Confirmar que el flood de alucinaciones PARÓ**

Dejar correr ~10 min en silencio, luego:
```bash
ssh kza 'tail -200 ~/logs/kza-metrics.jsonl | python3 -c "
import sys,json
n=sum(1 for l in sys.stdin if (json.loads(l).get(\"event_type\")==\"wake\"))
print(\"wake events en últimas 200 líneas:\", n)"'
```
Expected: drásticamente menos eventos wake (idealmente 0 sobre silencio). Antes: ~500/hora. **Este es el resultado que prueba que F2 mata el flood.**

### Task 0.2: DECISIÓN

- [ ] **Si la cadena funciona end-to-end y el flood paró:** la arquitectura está validada. Proceder a Fase 1 (entrenar "Nexa"). Dejar el escritorio en `engine: openwakeword` con `hey_jarvis` temporalmente (wake = "hey jarvis" hasta tener "Nexa"), o revertir a `whisper` si "hey jarvis" molesta en el hogar. **Documentar la decisión.**
- [ ] **Si la grabación/STT del comando falla** (p.ej. `_check_vad_completion` no cierra bien sin el WhisperWakeDetector): saltar a Fase 5 (hardening del endpointing) ANTES de entrenar "Nexa".

---

## Fase 1 — Setup del pipeline de entrenamiento openWakeWord

> openWakeWord entrena un clasificador chico sobre features del embedding de Google. Necesita: positivos sintéticos ("Nexa" en muchas voces), negativos (datasets de features precomputados que provee openWakeWord), y opcionalmente positivos reales (Fase 2). Output: `.onnx`.

### Task 1.1: Instalar dependencias de training + descargar negativos

**Files:**
- Read-only / instalación

- [ ] **Step 1: Crear venv de training aislado (no tocar el venv de prod)**

Run (server):
```bash
ssh kza 'cd ~ && python3.13 -m venv ~/wake_train_venv && \
  ~/wake_train_venv/bin/pip install -q openwakeword onnxruntime numpy scipy torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3'
```
Expected: instalación OK. (CPU torch alcanza para el clasificador chico; la síntesis usa Kokoro/Piper aparte.)

- [ ] **Step 2: Descargar los features de entrenamiento de openWakeWord**

Run:
```bash
ssh kza '~/wake_train_venv/bin/python -c "import openwakeword; openwakeword.utils.download_models()" 2>&1 | tail -3 && \
  ls -la ~/.local/share/openwakeword 2>/dev/null || find ~ -iname "*.npy" -path "*openwakeword*" 2>/dev/null | head'
```
Expected: modelos base + features de negativos descargados. Si el dataset de negativos completo (ACAV, cientos de GB) no está, usar los **features precomputados** que openWakeWord publica (~varios GB) — ver [docs de training](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb).

### Task 1.2: Generar positivos sintéticos de "Nexa" (Piper español + Kokoro)

**Files:**
- Create: `scripts/train_wake_word.py` (generación)
- Create: `data/wake_training/nexa_synth/` (output WAVs)

- [ ] **Step 1: Generar ~5.000 muestras sintéticas de "Nexa" con voces en español**

Usar [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) con una voz Piper `es_*` (rioplatense si hay, si no `es_ES`/`es_AR`), variando velocidad 0.8-1.3x y pitch. Si piper-sample-generator solo soporta inglés para el speaker-mixing, usar Kokoro (ya en el server, `http://localhost:8880`) generando "Nexa" en las voces español disponibles + augmentación de velocidad/ruido.

Comando de referencia (ajustar a la herramienta elegida):
```bash
ssh kza '~/wake_train_venv/bin/python scripts/train_wake_word.py generate \
  --phrase "Nexa" --n-samples 5000 --lang es \
  --out data/wake_training/nexa_synth/'
```
Expected: ~5.000 WAVs de "Nexa". **Nota de riesgo (investigación):** español solo-sintético da recall ~0.41 — por eso Fase 2 (voz real) es importante.

---

## Fase 2 — Grabar muestras reales de "Nexa" (DEPENDENCIA DEL USUARIO, bloqueante para calidad)

> **REQUIERE AL USUARIO.** El recall en español mejora mucho con voz real ponderada 3×. Sin esto, el modelo puede tener miss-rate alto.

### Task 2.1: Capturar grabaciones reales de "Nexa" en cada habitación

**Files:**
- Create: `data/wake_training/nexa_real/`

- [ ] **Step 1: El usuario graba ~30-50 "Nexa" variados**

Pedir al usuario que grabe (con el propio XVF3800 del cuarto, para que el audio matchee el dominio):
- 30-50 repeticiones de "Nexa" por cada persona del hogar
- Variando: distancia (1m, 3m, 5m), tono (normal, bajo, apurado), con/sin ruido de fondo
- A 16kHz mono, en `data/wake_training/nexa_real/<persona>/`

Script helper (a crear si ayuda):
```bash
ssh kza '~/wake_train_venv/bin/python scripts/train_wake_word.py record \
  --out data/wake_training/nexa_real/gabriel/ --n 40 --device <xvf3800_idx>'
```
Expected: ≥30 grabaciones reales por persona. **Bloquea la calidad del Task 3.2.**

---

## Fase 3 — Entrenar + evaluar el modelo "Nexa"

### Task 3.1: Entrenar el clasificador openWakeWord

**Files:**
- Modify: `scripts/train_wake_word.py` (train+export)
- Create: `models/wakeword/nexa.onnx`

- [ ] **Step 1: Entrenar (positivos synth + real ponderado 3× + negativos)**

Run:
```bash
ssh kza '~/wake_train_venv/bin/python scripts/train_wake_word.py train \
  --positives data/wake_training/nexa_synth/ \
  --positives-real data/wake_training/nexa_real/ --real-weight 3 \
  --steps 50000 --out models/wakeword/nexa.onnx 2>&1 | tail -20'
```
Expected: archivo `models/wakeword/nexa.onnx` (~400KB). Puede correr en CPU (clasificador chico) o GPU nocturna.

### Task 3.2: Evaluar recall y false-accepts (DECISION GATE)

**Files:**
- Create: `scripts/eval_wake_word.py`

- [ ] **Step 1: Medir recall sobre held-out de "Nexa" real + FA sobre silencio real del server**

Run:
```bash
ssh kza '~/wake_train_venv/bin/python scripts/eval_wake_word.py \
  --model models/wakeword/nexa.onnx \
  --positives data/wake_training/nexa_real_holdout/ \
  --negatives-audio <silencio/ruido real capturado del server, ej. 1h> \
  --threshold 0.5 2>&1 | tail'
```
Expected: imprime recall (% de "Nexa" detectados) y FA/hora (falsos disparos sobre silencio/ruido).

- [ ] **Step 2: DECISIÓN**

- **Si recall ≥70% y FA <1/hora @ threshold razonable:** proceder a Fase 4 (deploy).
- **Si recall <70%:** (a) capturar más voz real (Fase 2), (b) subir augmentación, o (c) **fallback a Porcupine** (entrena "Nexa" desde texto en la consola web en segundos; requiere integrar `.ppn` en `WakeWordDetector` — tarea adicional no detallada aquí, ~medio día). Decidir con el usuario.
- **Si FA alto:** subir threshold y re-medir recall a ese threshold (tradeoff).

---

## Fase 4 — Deploy del modelo "Nexa"

### Task 4.1: Configurar y activar "Nexa"

**Files:**
- Modify: `config/settings.yaml` (wake_word escritorio)

- [ ] **Step 1: Apuntar el modelo + threshold del Task 3.2**

En `config/settings.yaml`, bloque wake_word del escritorio:
```yaml
    engine: "openwakeword"
    model: "models/wakeword/nexa.onnx"   # ruta al modelo custom (detector.py acepta paths .onnx)
    threshold: 0.5                        # valor calibrado en Task 3.2
```
> `WakeWordDetector.load()` detecta `.onnx` por extensión y lo carga como custom model (`_load_custom_model`). Verificar que `predict()` incluye el modelo custom en su loop (línea ~143+); si el `predict` solo mira `self._oww_model` y no `self._custom_models`, agregar el branch (TDD: test que `detect()` dispara con el modelo custom).

- [ ] **Step 2: Preflight VRAM + restart + confirmar carga**

Run:
```bash
ssh kza 'nvidia-smi --query-gpu=memory.free --format=csv,noheader -i 1 && \
  systemctl --user restart kza-voice.service && sleep 10 && \
  journalctl --user -u kza-voice.service --since "1 min ago" | grep -iE "Modelo personalizado cargado|nexa|wake"'
```
Expected: `Modelo personalizado cargado: nexa`.

- [ ] **Step 3: Smoke test "Nexa" + medición de flood**

Decir "Nexa, prendé la luz del escritorio" → verificar acción. Dejar 30 min y medir wake events sobre silencio (debe ser ~0).

### Task 4.2: Métricas de éxito (rúbrica Alexa)

- [ ] **Step 1: Comparar antes/después**

Run tras ~1 día:
```bash
ssh kza 'python3 - << "PY"
import json
from collections import Counter
c=Counter(); acc=0; tot=0
with open("/home/kza/logs/kza-metrics.jsonl") as f:
    for l in f:
        try: d=json.loads(l)
        except: continue
        if d.get("event_type")!="wake": continue
        ts=d.get("@timestamp","")
        if ts < "2026-06-01": continue  # solo post-deploy
        tot+=1
        if d.get("matched"): acc+=1
print(f"post-deploy: wake_events={tot} matched={acc} ({100*acc/max(tot,1):.0f}%)")
PY'
```
Expected: wake events totales caen de ~2000/hora a decenas/día; tasa de aceptación sube de 1.4% a >70%.

---

## ✅ RESULTADOS F0 (ejecutado 2026-05-31, en vivo en el server)

**Validado:** openWakeWord corre en Python 3.13/ONNX; el audio fluye al detector (RMS+scores medidos con instrumentación temporal); el wake **dispara** (`hey_jarvis` picos 0.32-0.46 aislado); **el flood del wake murió** (0 `WhisperWake` sobre silencio tras flipear). Fix del gate `missing_wake` deployado en `main.py` (`_gate_wake_words = () si engine != whisper`) — verificado que no bloquea comandos sin "nexa".

**DOS GAPS reales descubiertos (refinan el plan):**
1. **Placeholder `hey_jarvis` demasiado débil en voz hispanohablante:** 0.46 aislado/claro, pero ~0.18 cuando se dice pegado al comando (piso de ruido AGC ~0.13). Acierta ~3/5. → **confirma la necesidad del modelo "Nexa" entrenado**; el placeholder no sirve para validar UX ni para uso diario.
2. **El STT del COMANDO también alucina sobre silencio** (hallazgo nuevo importante): al hacer pausa post-wake, la ventana de grabación captura silencio amplificado por el AGC → `FastWhisperSTT` (con vad_filter=True) transcribió **`'¡Gracias!'`** (el gate lo frenó por `filler_word`). Eliminar el flood del WAKE no alcanza si la captura del comando graba silencio. **Causa:** el path openwakeword **resetea el buffer en la detección y graba lo que viene después, sin prebuffer** → (a) pierde el arranque del comando si el wake dispara tarde, (b) graba la pausa/silencio → alucina. El `WhisperWakeDetector` viejo no tenía esto porque acumulaba el utterance entero.

**Estado del sistema:** REVERTIDO a `engine: whisper` + `threshold: 0.35` (decisión usuario 2026-05-31: el hogar usa "nexa" mientras se construye el fix offline; el flood vuelve temporalmente). detector.py limpio (instrumentación removida). Gate fix queda en main.py (no-op bajo whisper). Backups `*.bak.*` en el server.

---

## Fase 5 — Hardening de captura: prebuffer + utterance único (AHORA CRÍTICA, no condicional)

> F0 demostró que es obligatoria. El path openwakeword debe capturar **wake+comando como UN solo utterance limpio** (como Alexa y como hacía el WhisperWakeDetector), no "resetear y grabar después".

### Task 5.1: Ring prebuffer + captura continua

**Files:**
- Modify: `src/pipeline/multi_room_audio_loop.py` (RoomStream: agregar ring buffer; audio_callback; handler de detección ~361-416; `_check_vad_completion`)

- [ ] **Step 1:** Agregar un ring buffer continuo por RoomStream (`collections.deque(maxlen=~10-13 chunks de 80ms` = ~0.8-1.0s)) que se llena SIEMPRE en el `audio_callback` (también cuando `not listening`).
- [ ] **Step 2:** En la detección del wake (línea ~366), en vez de `rs.audio_buffer = []`, inicializar `rs.audio_buffer = list(prebuffer)` (incluye los ~0.8s previos → captura el arranque del comando aunque el wake dispare tarde, y la propia wake word).
- [ ] **Step 3:** Endpointing: que `_check_vad_completion` cierre cuando hay silencio real sostenido tras voz, con un `min_command_duration` para no cerrar sobre la pausa inicial; y un `max_utterance_s` de seguridad. Tunear `silence_threshold` (hoy 0.015) sabiendo que el AGC eleva el piso a ~0.13 RMS — probablemente subirlo o usar el VAD del XVF3800 (pyusb `DOA_VALUE.speech_active`, ver `docs/2026-05-31_DIAGNOSTICO...`).
- [ ] **Step 4:** Proteger el STT del comando contra silencio: `initial_prompt` corto/vacío + `hotwords` + considerar rechazar clips con ratio de voz < 20% (Silero sobre el clip corto, ya compatible). Esto ataca el "¡Gracias!" del comando.
- [ ] **Step 5:** Validar con el placeholder (decir "Hey Jarvis prendé la luz" natural, sin pausa) que el comando se transcribe limpio y llega a HA. Luego con "Nexa" (post-F3).

### Task 5.2: (Opcional) Gate del XVF3800 VAD on-chip
- [ ] Pre-gate `DOA_VALUE.speech_active` por pyusb antes de alimentar el wake/STT (0-GPU, opera pre-AGC). Ver `docs/research/2026-05-31_DIAGNOSTICO_ALUCINACIONES_Y_ROADMAP_ALEXA.md` §C/Prioridad-1.

---

## Fase 6 — Cleanup (tras estabilizar)

### Task 6.1: Remover deuda

- [ ] **Step 1:** Si "Nexa" openwakeword queda como default permanente, marcar el path Whisper-as-wake como legacy (no borrar aún — puede servir de fallback). Remover `StreamingSTT` (código muerto). Acortar `initial_prompt` global (se regurgitaba ×128). Documentar thresholds finales.

---

## Self-Review

- **Spec coverage:** F2 del roadmap = entrenar "Nexa" + flipear engine. Cubierto: F0 valida la cadena (ya cableada), F1 setup training, F2 voz real (usuario), F3 train+eval+decision gate, F4 deploy+métricas, F5 hardening condicional, F6 cleanup. ✅
- **Dependencias:** F0 reversible y prueba la arquitectura sin "Nexa". F2 es bloqueante de calidad (voz usuario). F3 tiene decision gate (recall ≥70% o Porcupine fallback). ✅
- **Riesgos explícitos:** (1) recall español solo-sintético ~0.41 → mitigado con voz real 3× + Porcupine fallback. (2) `predict()` de WakeWordDetector quizás no incluye custom models en su loop → verificar/TDD en Task 4.1. (3) endpointing sin WhisperWakeDetector → Fase 5 condicional. (4) VRAM/OOM al reiniciar → preflight en cada restart. ✅
- **No-placeholder:** los pasos de training referencian un `scripts/train_wake_word.py` a crear; su implementación exacta depende de la herramienta (piper-sample-generator vs Kokoro) — Task 1.2 lo resuelve. Los comandos de server son concretos.
