# Diseño: Robustez de transcripción del COMMAND PATH — "sólido como Alexa si escucha"

> **Fecha:** 2026-06-15 · **Rama:** `feat/nexa-command-detection-fixes`
> **Objetivo (usuario):** que la transcripción audio→texto del command path casi
> nunca falle *dado que el sistema escuchó* — al nivel de Alexa. Alcance acotado
> por el usuario a **transcripción de audio a texto** (no wake-detection, no NLU/routing).
> **Origen:** fin de semana de uso real con compañeros y pareja; comandos "no bien
> recibidos". Diagnóstico del usuario: la falla más común fue **"no respondió nada"**,
> y fue **"pareja, sin patrón"** (no correlacionada con far-field ni multi-voz).

## 1. Resumen ejecutivo

- **"No respondió nada" + "parejo" = falla sistemática + descarte silencioso, no acústica puntual.** Si la causa fuera far-field o multi-locutor, correlacionaría con esas condiciones; que sea pareja apunta a algo que degrada *toda* captura por igual y a una arquitectura que **se rinde sin avisar**.

- **Hallazgo central (verificado en código): el pipeline descarta en silencio en 3 de 4 puntos.** Solo un caso angosto avisa por voz. Con la *misma* tasa de aciertos del STT, esto hace que KZA se *sienta* mucho menos sólido que Alexa, que **nunca hace nada en silencio**. El "nada" parejo es, en gran parte, este descarte mudo.

- **Decisión tomada con el usuario: Camino A (software-first) + medición**, dejando B (señal/AGC) y C (modelo/LoRA) como fases siguientes **gateadas por los números** que produzca la medición. El feedback de "no entendí" será un **earcon** (sonido corto), no voz.

- **Por qué software-first:** ataca el "nada" (earcon) y recupera garble (temperature fallback) en días, sin tocar el mic de prod, sin romper <300ms en el caso fácil, y deja instrumentación para decidir B/C con datos en vez de a ciegas. Replica el orden de "máximo apalancamiento" de `docs/research/2026-06-13_TRANSCRIPCION_RIOPLATENSE_24-7_ANALISIS.md` (software/medir → señal → modelo).

## 2. Diagnóstico — el command path audio→texto, etapa por etapa

Pipeline (verificado): Mic XVF3800 (far-field 3-4m, AGC×8, beam ch1) → openwakeword `nexa.onnx` → **AmbientGuard** (STRICT/COOLDOWN) → captura (preroll 1s, endpoint 500ms, cap 3.5s, RMS pre-gate) → **faster-whisper large-v3-turbo** (es, beam5, temp0, sin fallback, `vad_filter=false`, int8_float16) → **CommandAcceptanceGate** (empty/noise/filler/compression) → routing/NLU.

### 2.1 El descarte silencioso (hallazgo central)

| Punto de corte | Archivo:línea | ¿Avisa al usuario? |
|---|---|---|
| AmbientGuard rechaza el wake (STRICT/COOLDOWN) | `src/pipeline/multi_room_audio_loop.py:286` | ❌ Mudo (`return False`) |
| CommandGate reject (empty/noise/filler/compression>2.2) | `src/pipeline/request_router.py:449-455` y `:760-766` | ❌ Mudo (`response=""`) |
| LLM router confianza baja (<0.6) | `src/pipeline/request_router.py:563` | ❌ Mudo (`response=""`) |
| LLM intent sin verbo evidenciado | `src/pipeline/request_router.py:588` | ✅ Voz: "No te entendí bien, ¿me lo repetís?" |

**Conclusión:** Alexa siempre ack-ea o reprompta; KZA descarta mudo en casi todos los caminos de duda. Esta es la diferencia #1 con Alexa y **no es de transcripción** — es de feedback. Es la causa directa del "nada" parejo.

### 2.2 Debilidades del audio→texto que alimentan esos descartes (ordenadas por probabilidad de "nada" sistemático)

1. **Señal far-field con AGC×8 (raíz física).** `PP_AGCMAXGAIN:[8.0]` (`config/settings.yaml:1102`) amplifica el piso de ruido junto con la voz a 3-4m → degrada *toda* captura por igual (encaja con "parejo"). Literatura "denoise hurts ASR" (40/40 configs, +46.6% WER) y la investigación previa lo marcan como la pérdida más temprana y probablemente la mayor. **Fuera de Camino A** (es Camino B), pero documentado como causa raíz.
2. **`temperature=0` sin fallback** (`src/stt/whisper_fast.py:171`). Apaga el re-decode adaptativo estándar de Whisper. Un segmento difícil que decodifica a vacío/garble no tiene segunda chance → vacío → CommandGate lo dropea mudo. **Palanca de software más barata.**
3. **Modelo `large-v3-turbo` int8** (`config/settings.yaml:140`). Destilado (4 capas decoder), 8× más rápido pero más débil en audio difícil/acentuado. Señales de confianza muertas: `no_speech_prob≈1e-10`, `avg_logprob` invertido (`src/stt/whisper_fast.py:32-37`, `src/nlu/command_gate.py:76`). **Fuera de Camino A** (es Camino C).
4. **`initial_prompt` sin voseo** (`config/settings.yaml:168`). Solo rooms/dominios; Whisper puede normalizar `prendé→prende`. Palanca léxica infrautilizada (con riesgo de acción-fantasma si se ponen frases-comando verbatim — incidente 2026-05-29).
5. **Captura cortada** (`max_utterance_s: 3.5`, `silence_end_ms: 500`). Comandos largos/con pausa se truncan → STT ve audio parcial.
6. **AmbientGuard mata wakes reales far-field** (commits del finde: "wake fuerte bypassa COOLDOWN", "capturar wakes RECHAZADOS"). Borde "si escucha", pre-transcripción; causa directa de "nada".

## 3. Camino A + medición — componentes

Los 5 componentes son **independientes y testeables por separado**, cada uno detrás de su flag en `config/settings.yaml` (cero archivos de config nuevos). Riesgo bajo, todo reversible, mantiene <300ms en el caso fácil.

### Componente 1 — Temperature fallback en `whisper_fast.py`

**Qué:** reemplazar `temperature=0` escalar (`whisper_fast.py:171`) por la lista `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, gateada **solo por `compression_ratio`** (única señal viva en turbo). El trigger por `logprob` se **desactiva** (`log_prob_threshold` muy bajo, p.ej. -3.0 o None) porque en turbo está invertido y dispararía mal; `no_speech_threshold` queda inocuo (no_speech degenerado).

**Cómo (faster-whisper):** `transcribe(..., temperature=[...], compression_ratio_threshold=<~2.0>, log_prob_threshold=<muy bajo>)`. faster-whisper re-decodifica subiendo temperatura mientras `compression_ratio > threshold`.

**Comportamiento:**
- Comando fácil → temp 0, comprime bajo → acepta primer pase → **0 latencia extra**.
- Comando garble far-field → comprime alto → re-decode 0.2/0.4… hasta texto no-repetitivo → **recupera** en vez de garble→drop→nada.
- **Sinergia con el gate de compresión** (enforce 2.2): el fallback *intenta arreglar*; el gate dropea lo que sigue siendo basura tras el fallback.

**Config:** `stt.temperature_fallback: {enabled: true, temperatures: [0.0,0.2,0.4,0.6,0.8,1.0], compression_ratio_threshold: 2.0, log_prob_threshold: -3.0}`. Defaults seguros; `enabled:false` revierte a temp 0 escalar.

**Honestidad sobre el alcance:** el fallback recupera **garble**, no necesariamente capturas verdaderamente vacías (silencio/no-voz). Para esas, la red es el earcon (Componente 2) y la medición (5).

### Componente 2 — Earcon "te oí, no te entendí"

**Qué:** un sonido corto (~200ms, p.ej. dos tonos descendentes) pre-cargado en RAM (extiende `src/tts/response_cache.py`, que ya cachea frases canónicas), disparado en los puntos hoy mudos: CommandGate reject (`request_router.py:453`) y confianza-baja LLM (`:563`). Reemplaza el `response=""` por reproducir el earcon. El reprompt de voz existente (`:588`) pasa a earcon también, por consistencia con la decisión "solo sonido" (configurable).

**⚠️ Decisión delicada — gating "humano plausible" (el earcon NO puede sonarle a la TV):**
- **Suena solo si** el wake fue de confianza alta **Y** la captura tuvo energía real (RMS sobre el piso) **Y** el reject fue `empty` / `high_compression` / `low_confidence`.
- **Nunca** suena en `noise_phrase` / `filler` / `word_repetition` (eso es TV/eco) ni en capturas de baja energía.
- Esto exige pasar `wake_score` + `rms` + `reason` hasta la decisión del earcon. `wake_score`/`rms` se conocen en captura (`_should_accept_wakeword`); el `reason` viene del `AcceptanceDecision`. El gating vive en un solo lugar testeable.

**Config:** `command_gate.earcon: {enabled: true, asset: "data/earcons/not_understood.wav", min_wake_score: 0.55, min_rms: 0.02, reasons: ["empty","high_compression","low_confidence"]}`. (`min_wake_score: 0.55` = wake fuerte, por encima del base 0.40; `min_rms: 0.02` arranca apenas bajo el piso inflado por el AGC ~0.025-0.05 — **calibrar ambos con repro** antes de confiar el gating.)

**Por qué importa:** un earcon que suena al azar/ante la TV te entrena a ignorarlo. Gateado a "humano plausible" es un "te oí y no te capté" genuino → es el mayor salto de solidez *percibida*, estilo Alexa, en el cambio más chico.

### Componente 3 — Voseo en `initial_prompt` (A/B, solo vocabulario)

**Qué:** agregar tokens de vocabulario rioplatense a `config/settings.yaml:168` — `vos, tenés, prendé, apagá, subí, bajá, poné, dale` — **como vocabulario, nunca como frases-comando** (por el incidente fantasma 2026-05-29 donde Whisper regurgitaba frases verbatim del audio ambiente). Detrás de un flag para A/B con/sin.

**Riesgo y mitigación:** regurgitación fantasma → mitigada por (a) solo-vocabulario, (b) el gate de compresión, (c) el temperature fallback. Validado con los contadores (Componente 5): comparar tasa de `fallback_recovered` y acciones-fantasma con/sin.

### Componente 4 — Endpointing (tweak medido)

**Qué:** subir `max_utterance_s` 3.5→5.0 (`config/settings.yaml`, bloque rooms) para no truncar comandos largos/con pausa. Detrás de flag, **validado con los contadores**, no a ciegas. `silence_end_ms` queda en 500 salvo que la medición muestre truncado por pausa.

### Componente 5 — Medición (convierte "parejo sin patrón" en datos)

**Qué:** contadores por room vía el `event_logger` existente → `data/events.db`. Eventos: `{accepted, gate_rejected:<reason>, low_confidence, earcon_fired, fallback_triggered, fallback_recovered}` con `room_id`, `wake_score`, `rms`, `compression_ratio`, `text[:60]`, timestamp.

**Para qué:** hoy no se puede *ver* la falla (db local vacía, fallas de comando no contadas). Con esto, el próximo finde: `SELECT reason, count(*) FROM ... GROUP BY reason` → se ve exactamente qué falló y en qué condición. **Esta tabla es el criterio de salto a Camino B (señal) o C (modelo).**

**Opcional (puente a Camino C):** persistir el clip de audio de las capturas rechazadas con su texto (patrón ya existente en `src/wakeword/wake_clip_writer.py`), para audit ciego y dataset de fine-tune.

## 4. Criterios de éxito y de salto a B/C

- **Éxito de A:** caída medible (en `events.db`) de `gate_rejected:empty`/`high_compression` y aparición de `fallback_recovered>0`; el usuario reporta que ya no hay "nada" mudo (suena el earcon). Sin regresión de acciones-fantasma ni de latencia p95 <300ms en comandos limpios.
- **Salto a Camino B (señal/AGC):** si tras A los contadores muestran garble/empty residual alto **parejo** (no correlacionado con reason de TV) → es señal far-field → A/B del AGC en banco (`PP_AGCMAXGAIN` Vía A / congelar gain Vía B), medido contra ground truth.
- **Salto a Camino C (modelo):** si tras A+B el garble persiste en audio *limpio* (no es señal) → `turbo→large-v3` y/o LoRA con el corpus capturado.

## 5. Riesgos

| Riesgo | Mitigación |
|---|---|
| Temperature fallback agrega latencia en casos duros | Solo se dispara en garble (compresión alta); el caso fácil queda a temp 0. Cap implícito por `max_utterance_s`. Medir p95. |
| Earcon suena ante la TV → ruido molesto | Gating "humano plausible" (wake alto + RMS + reason no-TV). Calibrar `min_wake_score`/`min_rms` con repro antes de subir umbrales. |
| Voseo en prompt reintroduce acción-fantasma | Solo-vocabulario + gate compresión + fallback; A/B medido; flag de rollback. |
| `max_utterance_s` 5.0 captura más charla ambiente | Detrás de flag; revertir si sube `gate_rejected:noise`. |
| Tocar el call site del gate (router) rompe el path legacy | Hay DOS call sites (`:449` y `:760`); cubrir ambos con tests. Gate ya es fail-open (`command_gate.py:96`). |

## 6. Fuera de alcance (explícito)

- Wake-word detection / AmbientGuard tuning (borde "si escucha", no transcripción). Se documenta como causa de "nada" pero no se toca en A.
- Camino B (AGC/firmware 6ch) y Camino C (modelo/LoRA): fases siguientes gateadas por la medición.
- NLU/routing más allá de los puntos de descarte silencioso.

## 7. Archivos a tocar (estimado)

- `src/stt/whisper_fast.py` — temperature fallback (Comp. 1).
- `src/tts/response_cache.py` + asset earcon — (Comp. 2).
- `src/pipeline/request_router.py` — disparo del earcon en `:453`/`:563`/`:588` con gating (Comp. 2).
- `src/nlu/command_gate.py` — exponer `reason` para el gating del earcon (Comp. 2).
- `src/pipeline/multi_room_audio_loop.py` — pasar `wake_score`/`rms` al earcon si hace falta (Comp. 2).
- `config/settings.yaml` — flags de Comp. 1-5; voseo en `initial_prompt` (Comp. 3); `max_utterance_s` (Comp. 4).
- `src/.../event_logger` (existente) — eventos de medición (Comp. 5).
- Tests en `tests/unit/...` por componente.
