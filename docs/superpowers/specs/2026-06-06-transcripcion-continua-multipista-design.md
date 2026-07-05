# Transcripción continua multi-pista sobre el XVF3800 — Diseño

**Fecha:** 2026-06-06
**Estado:** Aprobado por secciones en sesión de brainstorming (arquitectura, componentes, integración/fases)
**Rama:** `feat/nexa-command-detection-fixes`

## 1. Motivación y objetivos

El XVF3800 se usa hoy al mínimo: de los canales que puede exponer por USB se consume
uno solo (beam ASR), y Whisper transcribe únicamente la ventana post-wake. Mientras
tanto, el ambiente con TV satura el pipeline ("no me escucha") y todas las compuertas
acústicas están rotas o apagadas (SPENERGY=0, Silero prob~0 sobre audio post-DSP,
min_wake_rms=0).

Objetivos confirmados con el usuario (en orden):

1. **No perder comandos** — robustez con TV/ruido de fondo.
2. **Memoria ambiental (RAG)** — transcribir lo hablado en casa y destilarlo a memoria.
3. **Filtrar la TV** — distinguir voz en vivo de audio reproducido.

Requisitos de marco:

- **Pistas:** extraer del chip todos los canales físicos que exponga (decisión del
  usuario: "canales físicos crudos").
- **Recursos:** decisión final basada en benchmark (Fase 0), no en estimaciones.
- **Privacidad:** política **destilar y descartar** — el texto crudo vive solo en un
  buffer de horas; un proceso periódico extrae hechos a memoria y borra el literal.
  La destilación es **100% local** (no se manda audio ni texto crudo a ningún cloud).

No-objetivo de esta etapa: conversación sin wake word (follow-up ya existe y no se
amplía aquí).

## 2. Hallazgos que habilitan el diseño

1. **GPU 0 está libre** (verificado en el server 2026-06-06: 226 MiB / 8192 MiB).
   Desde que el reasoner migró a cloud, nadie reocupó cuda:0 → ~7.9 GB de VRAM
   disponibles + Threadripper casi ocioso.
2. **Existe firmware oficial de 6 canales**:
   `respeaker_xvf3800_usb_dfu_firmware_6chl_v2.0.x.bin` (Seeed). Layout:
   ch0=beam Conference (post-procesado), ch1=beam ASR (post-beamformer, sin
   post-proceso no lineal), ch2-5=mics 0-3 crudos pre-DSP. UAC2 estándar,
   16 kHz / 32-bit, sin driver propietario. Reversible por el mismo mecanismo DFU.
3. **Los mics crudos arreglan las compuertas rotas**: Silero VAD daba prob~0 porque
   opera sobre la salida post-DSP del chip; sobre mic crudo funciona. Con 4 mics
   crudos se puede calcular DoA propio (GCC-PHAT) y discriminar la TV por dirección.
4. **Correcciones a supuestos previos** (doc oficial XMOS):
   - SPENERGY se calcula "during post-processing" — **no** es un VAD pre-AGC; por eso
     "se rompió" al bajar MAXGAIN.
   - El AEC del chip está ciego en nuestro setup: la referencia far-end es una
     *entrada* que nadie le da (TTS/música salen por el MA1260; la TV jamás tendrá
     referencia). El filtro TV debe ser DoA + speaker-ID, no AEC.
5. **SOTA local viable**: Parakeet v3 (multilingüe con español, streaming, corre
   hasta en CPU), `diart` para diarización online, DeepFilterNet3 denoise en CPU.
   Whisper turbo continuo de 1 pista usaría ~5-10% de una 3070.

## 3. Arquitectura

```
XVF3800 (firmware 6ch, 16kHz/32-bit UAC2)
│
├─ ch1 (beam ASR) ──────────► COMMAND PATH (intacto, cuda:1)
│                              wake → STT comando → router → TTS  (<300ms)
│
├─ ch1 (tee) ───────────────► AMBIENT PATH (nuevo, cuda:0 + CPU)
├─ ch2-5 (mics crudos) ──┐
│                        ├──► Silero VAD (sobre crudo) → segmenta utterances
├─ ch0 (Conference) ──┐  ├──► DoA worker (GCC-PHAT, CPU) → azimut por utterance
│                     ▼  ▼
│              AmbientTranscriber (GPU 0)
│              STT continuo por utterance + speaker-ID + clasificador TV/live
│                     │
│                     ▼
│              AmbientUtterance(text, speaker, azimuth, source=live|tv|unknown,
│                               room, t0, t1, confidence)
│                     │
│            ┌────────┴─────────┐
│            ▼                  ▼
│     Buffer crudo TTL      Señales en caliente al command path:
│     (SQLite, horas,       • compuerta anti-TV para el wake
│      purga automática)    • segunda opinión si el comando salió garbleado
│            │
│            ▼ (cada N horas)
│     Destilador LLM local → hechos a memoria (ChromaDB) → borra crudo
```

Principios:

1. **El command path no se toca.** El ambient path es un consumidor más del stream
   multicanal, corre en cuda:0 y CPU. Si se cae o se atrasa, los comandos no se
   enteran. Kill-switch `ambient.enabled` en settings.yaml.
2. **No se transcribe silencio.** El VAD sobre mic crudo segmenta; el STT continuo
   solo procesa segmentos con voz. Con TV prendida el volumen de segmentos sube y el
   clasificador decide qué merece transcripción completa.
3. **Etiquetar, no filtrar a ciegas.** Cada utterance sale con `source`; `tv` muere
   en la purga del buffer, solo `live` se destila a memoria.

## 4. Componentes (módulo nuevo `src/ambient/`)

DI por constructor, `@dataclass` para DTOs, async/await para I/O — patrones del
proyecto.

| Componente | Qué hace | Dónde corre |
|---|---|---|
| `MultiChannelTap` | Se engancha al callback de `RoomStream` existente: el command path sigue leyendo ch1 como hoy; el tap encola ch0/ch1/ch2-5 en ring buffers lock-free para el ambient path. Buffers llenos → descarte FIFO (nunca bloquea el callback de audio). | Thread de audio existente |
| `UtteranceSegmenter` | Silero VAD sobre un mic crudo (ch2). Abre utterance al detectar voz, cierra a ~700 ms de silencio, pre-roll 0.5 s, tope 30 s por segmento. | CPU (~1% core) |
| `DoAWorker` | GCC-PHAT sobre pares de mics crudos durante cada utterance → azimut mediano + estabilidad. Calibración única del azimut de la TV por room (`ambient.tv_azimuth`). | CPU |
| `AmbientSTT` | Instancia separada del STT (Whisper turbo int8 o Parakeet v3 — lo decide la Fase 0) en `cuda:0`. Transcribe el segmento del **beam ASR (ch1)** correspondiente a la ventana que abrió el VAD sobre crudo (el beam tiene mejor SNR; el crudo solo segmenta/localiza). Cola de utterances; no compite con el Whisper de comandos de cuda:1. | GPU 0 (~1-2 GB) |
| `SpeakerTagger` | ECAPA-TDNN segunda instancia en cuda:0 → embedding por utterance → match contra usuarios enrolados (umbral 0.75, el mismo de `speaker_identifier`). Sin enrolar → `unknown`. | GPU 0 (~0.5 GB) |
| `SourceClassifier` | Reglas: DoA ≈ azimut TV + estable + hablante desconocido → `tv`; hablante conocido → `live`; resto `unknown`. Umbrales en settings. Shadow mode primero. | CPU |
| `AmbientStore` | SQLite `data/ambient.db`, tabla `utterances`, TTL `ambient.retention_hours` (default 12 h), purga horaria. Separada de `events.db`. | CPU |
| `Distiller` | Job asyncio cada N horas: lotes de utterances `live` → Qwen2.5-7B local (:8101) con prompt de extracción de hechos → escribe a `memory_db` (ChromaDB vía `LongTermMemory`) → marca destilado; la purga borra el crudo. | CPU + :8101 |

Presupuesto estimado en cuda:0 (a validar en Fase 0): STT ~1.5-2 GB + ECAPA ~0.5 GB →
quedan ~5 GB de margen.

### Decisiones de diseño registradas

- **Destilador local, nunca cloud**: mandar conversaciones del hogar a MiniMax
  contradice "destilar y descartar". Se usa el Qwen2.5-7B que ya corre en :8101.
  Aprobado por el usuario.
- **Segunda instancia de ECAPA/STT en cuda:0** en lugar de compartir las de cuda:1:
  aísla el ambient path del command path a costa de VRAM que hoy sobra.
- **SQLite dedicada (`ambient.db`)** en lugar de `events.db`: ciclo de vida distinto
  (TTL corto + purga agresiva vs analytics 90 d).
- **Esquema `AmbientUtterance`**: `text, speaker, azimuth, source, room, t0, t1,
  confidence` — t0/t1 absolutos para poder correlacionar con eventos del command path.

## 5. Integración con el command path

1. **Compuerta anti-TV para el wake**: cuando `nexa.onnx` dispara,
   `MultiRoomAudioLoop` consulta al ambient path si hay una utterance activa
   clasificada `tv` cubriendo ese instante; si sí, exige score estricto (reutiliza la
   escalera del `AmbientGuard`: STRICT situacional). **Shadow mode primero**, flip
   con datos.
2. **Segunda opinión post-comando**: si el gate de confianza marca el comando como
   garbleado, el router puede pedir al `AmbientStore` la transcripción ambiental de
   la misma ventana temporal (otra pista, otro STT) y comparar antes de rechazar.
3. **VAD crudo reemplaza a SPENERGY**: el `UtteranceSegmenter` expone "hay voz ahora"
   — disponible como pre-gate del wake, también shadow primero.

## 6. Fases de implementación

- **Fase 0 — Benchmark y validación (sin tocar producción):**
  `tools/benchmark_ambient.py` contra cuda:0 del server: RTF y VRAM de Whisper-turbo
  vs Parakeet v3 en español far-field; costo de ECAPA/Silero/GCC-PHAT. Con números se
  fija el modelo. Se descarga el firmware 6ch y se documenta procedimiento DFU +
  rollback.
- **Fase 1 — Flasheo 6ch (GATED):** requiere OK explícito del usuario en el momento.
  Ventana de bajo uso, firmware 2ch a mano, verificación post-flasheo (`arecord -c 6`,
  `xvf_controller` responde, kza-voice levanta). Falla → rollback inmediato.
- **Fase 2 — Ambient path en shadow:** stack completo etiquetando y persistiendo,
  sin afectar el command path ni destilar. Una semana de datos para calibrar
  `SourceClassifier` y azimut de TV.
- **Fase 3 — Flip con datos:** compuerta anti-TV enforced, destilador activo,
  señales en caliente conectadas.

## 7. Manejo de errores

- Ambient path **best-effort por contrato**: excepción en cualquier worker → log +
  reinicio del worker con backoff; nunca propaga al pipeline de voz.
- Sin GPU 0 disponible al boot → `ambient.enabled` efectivo = false con warning.
- Colas con tope: atraso → descarte FIFO del más viejo. Perder transcript ambiental
  es aceptable; bloquear el audio no.
- Firmware: si el device enumera con menos canales de los esperados (p. ej. rollback
  a 2ch), el tap degrada a modo 2ch automáticamente (sin mics crudos: VAD/DoA off,
  STT ambiental sobre ch0/ch1 sigue).

## 8. Testing

- TDD. Unit por componente en `tests/unit/ambient/` con fixtures de audio multicanal
  sintético; mocks de modelos en `tests/mocks/`.
- Integración del flujo tap → segmenter → STT → store con audio real grabado
  (muestras en `/tmp/nexa_debug/` del server).
- Benchmark de Fase 0 como script reproducible.
- La suite actual (633 tests) no se modifica.

## 9. Configuración (todo en `config/settings.yaml`, sección nueva `ambient:`)

```yaml
ambient:
  enabled: false            # kill-switch global (Fase 2 lo prende en shadow)
  stt_model: "whisper-v3-turbo"   # o parakeet-v3 según Fase 0
  stt_device: "cuda:0"
  retention_hours: 12
  distill_interval_hours: 6
  distill_llm_url: "http://127.0.0.1:8101/v1"   # SOLO local
  shadow_mode: true         # etiqueta y loguea, no afecta command path
  rooms:
    escritorio:
      tv_azimuth: null      # se calibra en Fase 2 (radianes)
      tv_azimuth_tolerance: 0.35
```

## 10. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Flasheo DFU falla en el único mic de producción | Fase 1 gated con OK explícito, firmware 2ch a mano, procedimiento de rollback documentado y probado en Fase 0 |
| El firmware 6ch se comporta distinto (tuning, xvf_host) | Verificación post-flasheo de `xvf_controller` y parámetros; degradación automática a modo 2ch |
| VRAM real en cuda:0 mayor a la estimada | Fase 0 mide antes de comprometer; margen de ~5 GB estimado |
| Clasificador TV con falsos positivos (bloquea comandos reales) | Shadow mode + una semana de datos antes del flip; el wake nunca se bloquea, solo se endurece el umbral |
| Crecimiento de `ambient.db` | TTL 12 h + purga horaria; solo texto (sin audio) |
| Destilador alucina hechos | Prompt de extracción conservador + confidence mínima por utterance + revisión vía dashboard (:9500) como follow-up |
