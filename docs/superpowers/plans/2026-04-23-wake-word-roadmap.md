# KZA Wake Word — Roadmap completo post-sesión 2026-04-23

**Estado**: Pipeline end-to-end funcional. Wake word "nexa" vía Whisper+VAD detecta y ejecuta comandos HA correctamente en ambiente silencioso. Falla en ambiente con TV. Este doc es el plan para llevarlo a robustez de producción.

---

## 1. Executive summary

El asistente KZA tiene un voice pipeline nativo corriendo como `kza-voice.service` que actualmente usa **WhisperWakeDetector** (STT-based wake word + trigger inline audio) en vez de OpenWakeWord porque el trainer oficial no está disponible. Funciona bien con ambiente silencioso, falla con TV de fondo. Tres mejoras en orden de impacto:

1. **Speaker diarization pre-wake** — filtrar audio por voz del usuario usando ECAPA-TDNN (ya cargado). Mata el problema de TV/otras voces.
2. **Streaming STT** — reducir latencia de detección end-of-utterance de ~800ms a ~150ms.
3. **Custom wake word neural (path C)** — modelo OpenWakeWord entrenado con piper-sample-generator + MUSAN. Máxima robustez, independiente de STT.

Trabajo total estimado: 8-15 horas divididas en 4 fases + validación.

---

## 2. Estado actual del sistema (2026-04-23, post-sesión)

### 2.1 Infraestructura operativa
- **kza-72b.service** (systemd --user, active): Qwen2.5-72B Q8_0 via `llama-cpp-python` server en `127.0.0.1:8200`, ~37GB RSS + mmap cache, linger=yes.
- **kza-voice.service** (systemd --user, active, enabled): voice pipeline nativo (excepción R10 #3 Notion pág 8 por USB ReSpeaker + MA1260 serial).
- **vLLM infra :8100** (externo, usuario `infra`): Qwen2.5-7B-AWQ, consumido por HTTP.
- **HA :8123** (externo): REST + WebSocket.
- **Layout**: `/home/kza/{app→kza,data,secrets,logs,.config/{systemd/user,containers/systemd}}`. Sub-rango puertos 9500-9599.

### 2.2 Pipeline de voz actual
```
Mic ReSpeaker device=9 (6ch, ch0 procesado AEC)
  → sounddevice InputStream (blocksize=1280 = 80ms)
  → multi_room_audio_loop.audio_callback
  → WhisperWakeDetector.detect(chunk)
     ├── VAD (silero-vad threshold=0.7)
     ├── RMS gate (min_rms=0.025)
     ├── Acumula utterance hasta silencio (500ms) o max (3.5s)
     ├── Whisper v3-turbo transcribe (cuda:0, int8_float16, lang=es)
     ├── Match substring "nexa"|"kaza" normalizado en primeras 3 palabras
     └── Si match → guarda audio post-wake en `_pending_command_audio` (offset ~400ms × palabras pre-wake)
  → multi_room_audio_loop detecta wake → pop_pending_command_audio()
     └── Si hay audio inline → lo usa DIRECTAMENTE como comando buffer (salta captura post-wake)
  → CommandProcessor: STT (Whisper) + SpeakerID (ECAPA-TDNN) + Emotion (wav2vec2)
  → RequestRouter:
     ├── NLU intent classifier (regex apagá/prendé → turn_off/turn_on)
     ├── Slot extractor (brightness_pct, rgb_color, color_temp_kelvin)
     ├── LastActionTracker (TTL 60s, toggle implícito para comandos ambiguos)
     ├── ChromaDB search_command con service_filter=intent, query_slots=merge
     └── await ha.call_service_ws(domain, service, entity_id, data)
  → ResponseHandler (Kokoro TTS)
```

### 2.3 Config relevante (`config/settings.yaml`)
```yaml
stt:
  model: "./models/whisper-v3-turbo"
  device: "cuda:0"
  compute_type: "float16"
embeddings:
  model: "BAAI/bge-m3"
  device: "cuda:0"
router:
  enabled: true
  base_url: "http://127.0.0.1:8100/v1"
  model: "qwen2.5-7b-awq"
reasoner:
  mode: "http"
  http_base_url: "http://127.0.0.1:8200/v1"
rooms:
  wake_word:
    engine: "whisper"
    words: ["nexa", "kaza"]
    threshold: 0.35  # solo para openwakeword legacy
    silence_end_ms: 500
    min_utterance_ms: 250
    max_utterance_s: 3.5
  escritorio:
    mic_device_index: 9
    bt_adapter: "hci0"
    # (entidades HA por definir)
```

### 2.4 GPU memory (runtime)
- **cuda:0** (~7.5GB usados): Whisper v3-turbo (~800MB) + Kokoro TTS (~1GB) + ECAPA SpeakerID (~500MB) + wav2vec2 Emotion (~1GB) + BGE-M3 (~1.5GB) + buffers (~2GB).
- **cuda:1** (~7.2GB usados): vLLM compartido (infra, external).
- **CPU**: Qwen 72B Q8_0 mmap.

### 2.5 Datos persistentes
- `data/chroma_db/`: 640 phrases indexadas (10 groups × ~15 specs × 4 frases).
- `data/wakeword_training/nexa/positive/`: 50 WAVs @ 2s ("nexa" varias entonaciones).
- `data/wakeword_training/nexa/negative/`: 80 WAVs @ 2s (hard negatives: "exa", "ne", "alexa", "taxi", "examen", "excelente" + frases TV-like).
- `data/wakeword_training/che_bo/positive/`: 8 WAVs legacy (borrable).
- `models/whisper-v3-turbo/`: modelo CT2 con preprocessor_config.json.
- `secrets/.env`: `HOME_ASSISTANT_URL`, `HOME_ASSISTANT_TOKEN`, `SPOTIFY_*`.

### 2.6 Archivos modificados en sesión actual (tracked)
- `src/llm/reasoner.py`: FastRouter → HTTP client; agregada HttpReasoner class.
- `src/llm/__init__.py`: exports actualizados.
- `src/main.py`: FastRouter HTTP constructor; HttpReasoner path (mode=http); WhisperWakeDetector factory (engine=whisper).
- `src/pipeline/request_router.py`: intent classifier integrado, slot extractor, LastActionTracker, fix `EventType.COMMAND` (no string).
- `src/pipeline/multi_room_audio_loop.py`: `pop_pending_command_audio()` integration; `logger.exception` con traceback.
- `src/vectordb/chroma_sync.py`: `search_command` con `service_filter` + `query_slots` + merge_service_data.
- `src/providers/factory.py`: FastRouter HTTP.
- `src/orchestrator/dispatcher.py`: **fix crítico** `call_service` → `call_service_ws` + await.
- `src/orchestrator/action_context.py` (nuevo): `LastActionTracker` con TTL.
- `src/nlu/{__init__.py,slot_extractor.py}` (nuevos): intent + extractor léxico.
- `src/stt/streaming_stt.py` (nuevo, no integrado aún): streaming VAD+Whisper wrapper.
- `src/wakeword/whisper_wake.py` (nuevo): `WhisperWakeDetector` con VAD gate + RMS gate + require_start + pending_command_audio.
- `src/wakeword/detector.py`: fix int16-scale para OpenWakeWord predict.
- `src/conversation/follow_up_mode.py`: fix `_schedule_timeout` con `threading.Timer` fallback cuando no hay event loop.
- `scripts/sync_ha_to_chroma.py`: auto-discovery de capabilities (onoff/brightness/color_temp/color/effect) + vLLM 7B via HTTP.
- `scripts/test_voice_command.py`: press-to-talk con intent filter + slot extractor.
- `scripts/train_{che_bo,nexa}.py`: recorder + trainer CLI (trainer oficial roto).
- `scripts/diagnose_wakeword.py`: multi-canal + escala testing.
- `config/settings.yaml`: todas las secciones actualizadas.
- Memoria `~/.claude/projects/-Users-yo-Documents-kza/memory/*.md`: entradas nuevas sobre arquitectura post-migración, GPU, servicios, convenciones.

### 2.7 Bugs conocidos NO bloqueantes
- `follow_up_mode._timeout_handler` coroutine warning (benigno — threading.Timer fallback funciona, pero la coroutine se crea y nunca await'a). Fix: `coro.close()` en el except.
- Otros `ha.call_service` sin await (dashboard/api.py:389, command_learner.py:523, routine_executor.py:201, intercom sí tiene await ok). No afectan el happy path actual pero deberían fixearse.
- `FeatureManager start failed: 'ZoneManager' has no attribute 'get_all_zones'` — warning de startup, no afecta runtime.
- `Zona no encontrada: zone_escritorio` al hacer call_service — ZoneManager busca por alias pero el lookup no está completo. No bloquea HA call.

---

## 3. Problemas identificados (por los que estamos acá)

| # | Problema | Impacto | Fase solución |
|---|---|---|---|
| P1 | Whisper transcribe audio de TV continuamente (VAD lo considera voz) → CPU waste + riesgo false positive si TV dice wake word | Crítico | Fase 1 |
| P2 | Whisper confunde "nexa" con "next up"/"negesa"/"un exa"/"aprender" — palabra novel + acento | Medio (causa miss ocasional) | Fase 3 |
| P3 | Trigger es end-of-utterance — latencia percibida ~800ms entre decir la palabra y el trigger | Bajo (UX, no funcional) | Fase 2 |
| P4 | Con TV tapando voz user, el audio inline del wake queda contaminado → STT post pifía el comando | Crítico | Fase 1 (resuelve parcialmente) |
| P5 | 50 positivas + 80 negativas grabadas sin usar (trainer openwakeword_utils roto) | — | Fase 3 |
| P6 | follow_up_mode warning del coroutine huérfano | Bajo (log noise) | Fase 4 |

---

## 4. Plan por fases

### Fase 1 — Speaker Diarization (2-3 hs)

**Objetivo**: pre-filtro por embedding de voz. Solo procesar audio que matchee el perfil vocal del usuario.

**Arquitectura**:
```
Audio chunk (80ms)
  → VAD (silero) → si is_speech
  → [NUEVO] ECAPA embedding(chunk acumulado ≥1s)
  → [NUEVO] cosine_similarity(embedding, gabriel_ref_embedding)
  → si sim >= threshold → procede a Whisper STT
  → si sim < threshold → descarta silencioso (log DEBUG)
```

**Tareas**:

**1.1 Script enrollment** (`scripts/enroll_voice.py`):
- Input: nombre user + directorio con WAVs (o listado). Por default: `data/wakeword_training/nexa/{positive,negative}/`.
- Cargar cada WAV → si dur >= 1s → pasar por ECAPA.get_embedding() → acumular.
- Si dur < 1s → padear con ceros o skip (ECAPA quality degrada <1s, mejor skip).
- Promedio de embeddings (L2-normalized mean).
- Guardar en `data/users/{user_id}_voice.npy` + metadata `.json` (n_samples, timestamp, model_version).
- Output: path del embedding + stats (cuántos samples usados, mean similarity entre samples = consistencia intra-user).

**1.2 Extender `WhisperWakeDetector`** (`src/wakeword/whisper_wake.py`):
- Nuevos params:
  ```python
  speaker_identifier: SpeakerIdentifier | None = None  # ECAPA instance
  speaker_embedding: np.ndarray | None = None          # ref embedding
  speaker_threshold: float = 0.65                       # similarity cutoff
  speaker_min_audio_s: float = 0.8                     # ECAPA quality floor
  ```
- Método `_speaker_match(audio: np.ndarray) -> bool`:
  - Si speaker_embedding is None → return True (filtro desactivado)
  - Si dur < speaker_min_audio_s → return True (no confiar en embedding de audio corto)
  - Compute embedding → cosine similarity vs ref → return sim >= threshold
  - Log DEBUG: `"Speaker check: sim={sim:.3f} {'PASS' if match else 'REJECT'}"`
- En `_process_chunk`: antes de llamar `_transcribe_and_match`, si hay speaker_filter, hacer `_speaker_match`. Si falla → return None, skip Whisper.
- Beneficio secundario: se ahorra el STT de cada utterance de TV (~200ms × N utterances/min = big CPU save).

**1.3 Config** (`config/settings.yaml`):
```yaml
rooms:
  wake_word:
    speaker_filter:
      enabled: true
      embedding_path: "./data/users/gabriel_voice.npy"
      threshold: 0.65
      min_audio_s: 0.8
```

**1.4 Wire-up** (`src/main.py`):
- Pasar al WhisperWakeDetector: `speaker_identifier=command_processor.speaker_identifier` (ya cargado).
- Cargar embedding: `np.load(embedding_path)` si existe + enabled.
- Validar al startup que el embedding tiene la dim correcta del modelo ECAPA.

**1.5 Validación** (checkpoints):
- Con TV prendida + user NO habla: log debe mostrar filtered speakers `sim ~0.3-0.5`. Cero Whisper transcribes.
- Con TV prendida + user dice "Nexa, prendé luz escritorio": log muestra filtered TV + `sim ~0.80-0.95` + 🔥 trigger + HA exec.
- Con TV apagada + user dice "Nexa...": 🔥 trigger normal.
- Medir: reducción de llamadas Whisper (debería caer a <1/min con TV típica).

**1.6 Edge cases**:
- User con resfriado / cambio de entonación → threshold baja. Solución: re-enroll periódicamente o bajar threshold a 0.55.
- User habla muy corto ("Nexa"): <1s, pasa por skip → no filtra. OK, el require_start del substring match sigue filtrando.
- Otro miembro familia (ej hermana) también quiere usar: multi-user enrollment. El UserManager ya soporta; el filtro debería aceptar match contra CUALQUIER de N embeddings enrolados.

**1.7 Archivos a crear/modificar**:
- CREAR: `scripts/enroll_voice.py`
- MODIFICAR: `src/wakeword/whisper_wake.py`, `src/main.py`, `config/settings.yaml`
- OPCIONAL: actualizar `src/users/user_manager.py` si queremos extender para persistir embeddings activos en JSON.

---

### Fase 2 — Streaming STT (3-4 hs)

**Objetivo**: reducir latencia del trigger end-of-utterance → word-synchronous (~100-200ms).

**Contexto**: `faster-whisper` >= 1.0 soporta streaming con `word_timestamps=True`. La idea es no esperar silencio total — re-transcribir cada ventana de 1.5s con overlap 500ms y buscar wake word en los resultados incrementales.

**Arquitectura**:
```
Audio stream continuo
  → ring buffer 2s (deque samples)
  → cada 200ms:
    → if VAD active last 500ms:
      → transcribe ring buffer (Whisper word_timestamps=True)
      → if wake word aparece en últimos 500ms del transcript:
        → trigger inmediato (timestamp de la palabra)
        → recortar audio desde wake_end_time → pass as command
```

**Tareas**:

**2.1 Refactor `WhisperWakeDetector`** a modo streaming:
- Opción A: modificar clase actual con flag `streaming=True` que cambia el behavior del `_process_chunk`.
- Opción B: crear clase hermana `StreamingWhisperWakeDetector` que reemplaza la actual cuando config `rooms.wake_word.streaming=true`.
- Recomendada: Opción B. Mantiene la clase actual como fallback "seguro", menos invasivo.

**2.2 Implementación**:
- Ring buffer: `collections.deque(maxlen=fixed_samples)` con lock para thread safety.
- Timer/scheduler: cada 200ms desde último analyze, si VAD en estado "active" en algún chunk de los últimos 500ms, lanzar Whisper sobre el ring buffer.
- Whisper call con `word_timestamps=True`, `condition_on_previous_text=False` (evita drift).
- Parse segments → buscar wake word en words con `start_time >= (now - 500ms)`.
- Si match: calcular `wake_end_time` del word matched. Extraer `audio[wake_end_time:]` del ring buffer como `pending_command_audio`.
- Dedup: solo trigger si `now - last_trigger > 2s` (evita doble trigger con overlap).

**2.3 Integración pipeline**:
- Multi-room audio loop YA llama `detect(chunk)` cada 80ms. El streaming detector acumula internamente y decide cuándo transcribir. API externa (`detect() → tuple|None`) no cambia.
- `pop_pending_command_audio()` ya existe, mantiene contrato.

**2.4 Latencia objetivo**:
- Trigger: wake_word pronunciado → detect() returns ≤150ms.
- Command STT start: inmediato (ya tenemos audio post-wake del ring buffer).
- End-to-end objetivo: <400ms wake→ejecución HA.

**2.5 Trade-offs**:
- CPU/GPU: Whisper cada 200ms vs cada ~3.5s → ~15× más inferencias. Whisper v3-turbo hace ~200ms por ventana de 2s → ~1 inferencia activa continuamente. ~30-50% GPU:0 sostenido (antes <5%). Medir y decidir si vale la pena.
- Word timestamps: agrega ~30% a la latencia del Whisper call. Con turbo: ~250ms por ventana.
- **Mitigante**: combinar con Fase 1 (speaker filter). Solo transcribe cuando es el user → la mayoría del tiempo (TV) sale gratis.

**2.6 Archivos a crear/modificar**:
- CREAR: `src/wakeword/streaming_whisper_wake.py`
- MODIFICAR: `src/main.py` (factory según config), `config/settings.yaml`.

---

### Fase 3 — Custom Wake Word Neural (Opción C, 4-6 hs)

**Objetivo**: wake word independiente de STT, triggering mid-word, robusto a ambiente ruidoso.

**Contexto**: tenemos los 130 samples reales pero el training oficial de OpenWakeWord requiere ~10-20k samples sintéticos + noise corpus. El flow oficial es el notebook `openwakeword/notebooks/training_custom_models.ipynb`.

**Requisitos**:
- `piper-sample-generator` (pypi): TTS para generar miles de voces sintéticas diciendo "nexa".
- MUSAN corpus (~11GB): dataset open-source de speech/music/noise para negativas.
- TensorFlow 2.x, audiomentations.
- GPU idealmente (cuda:0 disponible ~30 min durante training).

**Tareas**:

**3.1 Setup**:
- Instalar deps en venv: `pip install piper-tts piper-sample-generator tensorflow audiomentations`.
- Descargar MUSAN: `wget https://www.openslr.org/resources/17/musan.tar.gz` (~11GB en `/home/kza/data/musan/`).
- Clonar repo openwakeword + notebook.

**3.2 Generación sintética**:
- Usar piper-sample-generator para producir 10k-20k WAVs de "nexa" con distintas voces (120+ voces españolas disponibles), velocidades, entonaciones.
- Augmentar con audiomentations: pitch shift, time stretch, add noise, reverb.
- Augmentar específicamente con muestras de MUSAN TV/music como ruido de fondo (match tu ambiente real).

**3.3 Training**:
- Seguir notebook oficial (con tweaks):
  - Input: 1.5s audio windows @ 16kHz.
  - Extract 96 mel features via openwakeword.utils.AudioFeatures.
  - Dataset: 70% synthetic + 30% real (tus 50 positivas) → ~3× augmented con noise → final ~30k positives.
  - Negatives: MUSAN + tus 80 hard negatives.
  - Split: train/val 80/20.
  - Model: 2-3 layer CNN + FC, <1M params.
  - Train 50-100 epochs, early stop on val_F1.
- Export `.onnx` en formato compatible openwakeword.Model custom path.

**3.4 Integración**:
- Copiar `.onnx` a `/home/kza/app/models/wakeword/nexa.onnx`.
- Config: `rooms.wake_word.engine: "openwakeword"`, `rooms.wake_word.model: "nexa"`, `custom_models_dir: "./models/wakeword"`.
- El `WakeWordDetector` existente (con el fix int16-scale aplicado) debería cargarlo via `custom_models_dir`.

**3.5 Validación**:
- Test con TV prendida: el modelo custom debería ignorar TV (fue entrenado con noise de fondo similar).
- Test precisión: medir false positives por hora + false negatives.
- Target: <5 false positives/día, <10% false negatives.

**3.6 Fallback strategy**:
- Config toggle permite switchear rápido entre whisper / openwakeword. Si modelo custom genera muchos FP → revertir a whisper+speaker_filter.
- Mantener WhisperWakeDetector como alternativa soportada.

**3.7 Archivos a crear/modificar**:
- CREAR: `scripts/train_custom_wake.py` (wrapper del notebook en código Python corrido server-side), `data/musan/` (dataset), `models/wakeword/nexa.onnx`.
- MODIFICAR: `config/settings.yaml` (switch engine).

---

### Fase 4 — Cleanup & Integration (1-2 hs)

**4.1 Fix bugs secundarios**:
- `follow_up_mode._schedule_timeout`: cerrar la coroutine cuando cae al fallback thread.Timer:
  ```python
  coro = self._timeout_handler(timeout)
  try:
      self._timeout_task = asyncio.create_task(coro)
  except RuntimeError:
      coro.close()  # ← agregar
      # ...timer fallback
  ```
- Auditar `ha.call_service` sin await:
  - `src/dashboard/api.py:389` (probablemente se corrige aceptando sync endpoint o await + async endpoint)
  - `src/training/command_learner.py:523`
  - `src/routines/routine_executor.py:201`
- Verificar ZoneManager: `zone_escritorio` no encontrado → investigar mapping zones vs rooms.
- FeatureManager warning `'ZoneManager' object has no attribute 'get_all_zones'`.

**4.2 Docs & memoria**:
- Update `docs/notion_page8_kza_update.md` con arquitectura final del wake word pipeline.
- Actualizar memoria `~/.claude/projects/.../memory/`:
  - Feedback sobre `call_service` vs `call_service_ws` vs `call_service_ws` + await.
  - Nueva entry sobre speaker filter + threshold tuning.
  - Gotcha: OpenWakeWord audio input escala int16.
- Pegar el update al Notion pág 8 §"Proyectos registrados".

**4.3 Tests**:
- Unit tests para WhisperWakeDetector con speaker filter (mocked ECAPA).
- Integration test end-to-end: audio WAV → trigger → HA mock call.
- Smoke test del pipeline completo que el user pueda ejecutar post-cambios.

---

## 5. Checkpoints / Decisiones de validación

Después de cada fase, antes de avanzar a la siguiente, validar:

### Post-Fase 1 (Speaker filter)
- [ ] Enrollment produce embedding con shape (192,) o (256,) según modelo ECAPA.
- [ ] Log muestra DEBUG "Speaker check: sim=X.XX" por cada utterance VAD.
- [ ] Con TV prendida 2 min sin user hablar: 0 transcripciones Whisper (antes: ~30).
- [ ] User dice "Nexa, apagá la luz" con TV prendida: trigger dispara, luz cambia.
- [ ] User dice nada por 1 min con TV prendida: CPU GPU:0 cae a <10% (antes: ~40% por Whisper continuo).

### Post-Fase 2 (Streaming)
- [ ] Latencia medida: user dice "Nexa" → log trigger. Target <200ms.
- [ ] Latencia end-to-end: user dice comando → luz física cambia. Target <500ms.
- [ ] CPU/GPU estable bajo carga continua.
- [ ] No duplicates triggers dentro de 2s.

### Post-Fase 3 (Custom neural)
- [ ] Modelo .onnx cargado sin error.
- [ ] Trigger con TV prendida, mid-word, confiable.
- [ ] False positive rate <5/día en uso normal.
- [ ] Se puede switchear entre whisper/openwakeword via config sin restart.

### Post-Fase 4 (Cleanup)
- [ ] Sin warnings en `journalctl -u kza-voice -p warning`.
- [ ] Todos los paths async correctamente await'd.
- [ ] Tests pasan `pytest tests/ -k wakeword`.

---

## 6. Preguntas abiertas / decisiones pendientes

1. **Multi-user enrollment**: ¿el pipeline soporta múltiples voces (Gabriel + familia) enroladas? UserManager lo permite. Si sí, el speaker filter debería aceptar match contra _cualquiera_ de los N enrolados, no solo 1.
2. **Threshold speaker filter**: arranco con 0.65. Ajustable con logs DEBUG empíricos.
3. **Reenrollment periódico**: ¿cada cuánto re-promediar embedding con audios nuevos del user? Recomendación: cada 30 días o cuando precision baje.
4. **Streaming STT — impacto GPU**: si cuda:0 se satura con streaming Whisper continuo, posible que sea necesario downgradear a Whisper small o medium. Medir primero.
5. **Custom neural — ventanas de 1.5s vs 2s**: openwakeword default es 1.5s. Para "nexa" que es corto (~400ms), 1s window puede alcanzar y reducir latencia.
6. **Nombres de wake word**: ¿queda "nexa" o cambiamos? "nexa" presenta riesgo acústico con "alexa" si la familia tiene un Echo. Alternativa: entrenar "kaza" (único, sin colisión).
7. **Qué entrena primero en Fase 3**: ¿solo nexa, solo kaza, o ambos? Ambos permite redundancia pero 2× el training time.

---

## 7. Orden recomendado de ejecución

**Sesión próxima** (cuando se retome):
1. Validar que el pipeline sigue funcional (5 min).
2. Ejecutar Fase 1 completa (speaker filter). Es el que más ROI por esfuerzo. (2-3 hs)
3. Validar con TV real.
4. (Opcional) arrancar Fase 2 si hay tiempo.

**Sesión N+1**:
- Fase 2 (streaming).
- Validación con métricas de latencia.

**Sesión N+2** (fin de semana, ≥4hs):
- Fase 3 (custom neural).
- Setup de piper-sample-generator, MUSAN, training notebook.

**Sesión final**:
- Fase 4 cleanup + tests + docs Notion.

---

## 8. Dependencias entre fases

```
Fase 1 (Speaker filter) ─┐
                         ├─→ Fase 4 (cleanup docs)
Fase 2 (Streaming) ──────┤
                         │
Fase 3 (Custom neural) ──┘
```

Fases 1-3 son **independientes** — se pueden hacer en cualquier orden. Mi recomendación es 1→2→3 por ROI decreciente. Fase 4 es el último (integra todo en docs).

---

## 9. Comandos de referencia rápida

### Estado del sistema
```bash
ssh kza@192.168.1.2 "systemctl --user status kza-voice.service kza-72b.service --no-pager"
ssh kza@192.168.1.2 "journalctl --user -u kza-voice.service -f"   # logs live
ssh kza@192.168.1.2 "nvidia-smi"                                   # GPU usage
```

### Test manual wake word
```bash
ssh -t kza@192.168.1.2 "cd /home/kza/app && systemctl --user stop kza-voice && .venv/bin/python scripts/train_nexa.py test --threshold 0.1"
# Ctrl-C y reinicia con: systemctl --user start kza-voice
```

### Probar comando end-to-end
Decir al mic del escritorio: "Nexa, apagá la luz del escritorio" o "Nexa, prendé la luz del escritorio"

### Rebuild chroma (si cambia el sync)
```bash
ssh -t kza@192.168.1.2 "cd /home/kza/app && .venv/bin/python scripts/sync_ha_to_chroma.py --wipe"
```

### Deploy código desde local
```bash
scp /Users/yo/Documents/kza/src/wakeword/whisper_wake.py kza@192.168.1.2:/home/kza/app/src/wakeword/
ssh kza@192.168.1.2 "systemctl --user restart kza-voice.service"
```

### Git status
```bash
cd /Users/yo/Documents/kza && git status
# Muchos archivos modificados en esta sesión. Considerar commit intermedio antes de más cambios.
```

---

## 10. Resumen histórico (contexto para retomar)

**Sesión 2026-04-21**: Migración arquitectural completa (layout estándar, vLLM HTTP consumer, HttpReasoner 72B service, NLU intent+slots, 640 phrases indexed, streaming_stt.py creado, ~47GB cleanup). 13 sprints completados.

**Sesión 2026-04-23 (actual)**:
- Intento OpenWakeWord custom training → trainer oficial roto (`openwakeword.utils.train_custom_model` no existe).
- Pivot a WhisperWakeDetector (STT-based).
- Debug int16-scale bug en OpenWakeWord runtime.
- Integración engine=whisper.
- Recording sesión: 50 positivas + 80 negativas de "nexa" (preservadas en disco).
- Problema TV: VAD y Whisper saturados con diálogos.
- Ajustes: VAD 0.7, RMS 0.04→0.025, require_start True.
- Fix crítico: `dispatcher.py:478` sin await → `call_service_ws` + await.
- Fix: `follow_up_mode._schedule_timeout` threading.Timer fallback.
- Fix: `EventType.COMMAND` enum (no string) en request_router.
- Fix: `WhisperWake pending_command_audio` — recorta audio post-wake para que pipeline no necesite recapturar.
- **Luz física prendió** con "Nexa, prendé la luz del escritorio" ✅.
- **Pendiente**: todo lo listado en Fases 1-4 arriba.

---

_Plan redactado 2026-04-23 antes de compactar contexto. Para retomar: leer este doc + memoria persistente `~/.claude/projects/-Users-yo-Documents-kza/memory/MEMORY.md` + git log del repo. El pipeline está OPERATIVO end-to-end en ambiente silencioso; la Fase 1 es el siguiente paso de alto impacto._
