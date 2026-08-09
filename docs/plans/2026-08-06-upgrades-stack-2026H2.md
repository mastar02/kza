# Plan de upgrades del stack — 2026 H2

**Fecha**: 2026-08-06
**Origen**: barrida de investigación (5 agentes, estado del arte agosto 2026) sobre STT, TTS, speaker ID, NLU router y wake word, cruzada contra el estado real de `config/settings.yaml` y los NO-GO medidos del proyecto.
**Restricciones vigentes**: cuda:1 a ~7.4/8GB (margen casi nulo), fast path <300ms, español rioplatense, doctrina de eval con held-out asimétrico definido ANTES de mirar resultados, server es producción real (consultar antes de tocar).

---

## Decisión 0 — ¿Sacar Whisper y quedar 100% Parakeet-TDT-0.6B-v3?

**NO por ahora. Se mantiene el veto híbrido.** Fundamentos:

1. El flip directo `stt.engine → parakeet` en el command path es un **NO-GO medido** (2026-07-14, garble/inglés en comandos). La solución vigente es el veto híbrido: Whisper primario + Parakeet en shadow como veto anti-alucinación, 0 latencia extra (Parakeet ~87ms < Whisper ~179ms). Está en `config/settings.yaml:176-195`.
2. El ambient path **ya corre Parakeet v3** desde 2026-06-07 (`ambient.stt.engine: parakeet`, `nemo-parakeet-tdt-0.6b-v3` vía onnx-asr en CPU). No hay upgrade pendiente ahí.
3. La investigación confirma que el problema del ambient es **SNR, no modelo** — sacar Whisper no arregla nada medido y pierde: la infra de LoRA semanal, el `initial_prompt` de dominio, y el motor que hoy gana en calidad sobre comandos.

**Camino data-driven para revisitarlo**: el `shadow_engine: parakeet` ya loguea el A/B en vivo. Analizar esos logs acumulados (Whisper vs Parakeet v3 sobre comandos reales en español) es gratis. Si Parakeet v3 empata o gana en el bucket de comandos con buen SNR, se rediscute con datos. Hasta entonces, Whisper se queda.

---

## Fases

### F0 — Quick wins (bajo riesgo, esta semana)

**F0.1 — Análisis de logs del shadow A/B Whisper vs Parakeet** (command path)
- Los datos ya existen (shadow activo). Extraer: tasa de acuerdo, casos donde difieren, calidad por bucket de vad.
- Salida: respuesta cuantitativa a la Decisión 0 + baseline para cualquier cambio de STT.
- Costo: una sesión de análisis. Riesgo: cero (read-only).

**F0.2 — Actualizar build de llama.cpp del :8101**
- FlashAttention y CUDA Graphs ahora son default en CUDA (hasta ~1.2x en batch=1, el caso exacto del router).
- ⚠️ Trampa conocida: FA + KV cache cuantizado cae **silenciosamente** a CPU si el build no tiene `GGML_CUDA_FA_ALL_QUANTS=ON`. Verificar tok/s antes/después con benchmark, no asumir (doctrina proxies mentirosos).
- Server es producción: coordinar ventana, `nvidia-smi` antes de reiniciar (VRAM apretada, riesgo doble-load OOM).
- Costo: 1-2h. Rollback: binario anterior.

**F0.3 — Spike: word boosting para Parakeet ambient**
- La investigación vendía NeMo 2.5 GPU-PB (boosting de "living"/habitaciones sin reentrenar), PERO el ambient corre Parakeet vía **onnx-asr en CPU**, que no lo soporta. Opciones a evaluar:
  a. sherpa-onnx con hotwords (prefix-tree, requiere transducer + `modified_beam_search`) — ¿soporta el TDT v3 offline? ¿latencia del beam vs greedy en clips 1-4s?
  b. NeMo runtime con `malsd_batch` — implicaría GPU (¿cuda:0?) y stack NeMo pesado. Cambio de asignación de GPU ⇒ se discute primero.
- Go/no-go: si (a) funciona con latencia comparable, adoptar; si solo queda (b), evaluar costo/beneficio aparte.
- Nota: el bug living→Libby (43%) es de **Whisper en el command path**, y el `initial_prompt` ya incluye "el living" — el prior léxico no alcanzó. El boosting de Parakeet ataca el ambient path; para el command path el fix real es F2.1 (LoRA ruidoso con las palabras problema en el corpus).

**F0.4 — Parakeet ambient y el español (verificado 2026-08-06)**
- El deploy está bien: server = main (md5 idéntico), config correcta, onnx-asr 0.11.0.
- Pero `language="es"` en `parakeet_stt.py` es **cosmético**: en onnx-asr solo la clase `NemoConformerAED` (Canary) usa el kwarg; `NemoConformerTdt` lo ignora. Y no es un límite de onnx-asr: **Parakeet TDT v3 no admite forzar idioma por diseño** (solo autodetección; NVIDIA remite a Canary para eso). El drift a inglés ocurre en audio de SNR bajo — consistente con los buckets medidos (vad>0.80 → 94% español; vad<0.20 → 87% garble inglés).
- Acciones: (a) ✅ corregido el docstring mentiroso de `parakeet_stt.py` (`8dfa061`); (b) ✅ **gate por vad en el wake textual** implementado (`4eed060`): `min_vad: 0.50` en `ambient.textual_wake`, calibrado contra ambient.db (garble "Next up." en vad 0.36-0.49, comandos reales desde ~0.57), fail-open con vad None, SOLO vad — jamás idioma. El distiller ya gateaba (0.45 + lang_ok); el wake textual era el único puente ambient→acción sin gate. **Pendiente deploy: coordinar POST-campaña keep_audio** (config del server tiene drift declarado hasta ~08-08); (c) opción a benchear: `nemo-canary-1b-v2` en el MISMO runtime onnx-asr (loader ya lo soporta, y ahí `language="es"` sí funciona) — costo: 1B AED autoregresivo en CPU, medir RTF antes de considerar. Solo si post-gate el bucket de vad alto sigue mostrando drift (hoy: 94% español ahí).

### F1 — Router: A/B Qwen3-4B-Instruct-2507 (alto retorno)

- Candidato: `Qwen3-4B-Instruct-2507` Q4_K_M (~2.5GB, **libera ~2GB de cuda:1**), non-thinking puro, mejor base medida para fine-tuning (pega en el QLoRA nocturno).
- Método: reusar el held-out + runner del proyecto clima (quedaron reusables). Held-out con casos de negación incluidos. Criterio go/no-go **asimétrico definido antes de correr**: cualquier consulta→acción nueva = NO-GO automático (lección del clima).
- Si GO: swap en :8101, re-apuntar el QLoRA nocturno a la base nueva, benchmark de latencia (`tools/benchmark_latency.py`).
- La VRAM liberada NO se reasigna sin discutir (regla del proyecto). Queda como margen — hoy el margen es el recurso más escaso.

### F2 — Datos: campaña keep_audio (desbloquea 3 frentes)

La campaña ya planificada en el proyecto de fidelidad ambient pasa a tener **tres clientes**:

**F2.1 — LoRA de Whisper con audio ruidoso far-field propio**
- Evidencia dura: fine-tunear con audio limpio NO mejora el WER ruidoso (Whisper-Flamingo: con ruido 20.8%→11.7%). El LoRA semanal actual (base `distil-whisper-large-v3-es`) debe entrenar con far-field real de las habitaciones + las palabras problema ("living", nombres) en el corpus.
- Este es el fix estructural de living→Libby en el command path.

**F2.2 — Cohorte para speaker ID** (ver F3)

**F2.3 — Sets de eval far-field** para cualquier A/B futuro de STT (incluida la Decisión 0).

### F3 — Speaker ID: backend primero, modelo después

Hallazgo central: el SOTA far-field con todo el arsenal ronda 3-6% EER; el 0/5.939 con enrollment near-field está **fuera del régimen donde cualquier modelo funciona**. Orden obligado:

1. **Re-enrollment far-field**: sesión de ~10 min/persona con el ReSpeaker de techo, varias posiciones. Promediar embedding near+far por usuario.
2. **AS-norm + Sub-Mean** con cohorte del propio hogar (audio de TV incluido, sale de F2.2).
3. **Gate por bucket de vad** (el instrumento del proyecto fidelidad ya lo mide): no intentar identificar sobre audio que ni el STT transcribe. A vad<0.20 el speaker ID es irrecuperable — no es un bug, es física.
4. **Re-correr la eval de 5.939 muestras** con 1-3. Si la distribución sigue plana ⇒ ningún modelo lo arregla, parar acá.
5. Solo si 4 muestra señal: A/B de modelo — **CAM++ ONNX** (29MB, sherpa-onnx CPU, una tarde) y **ReDimNet B2/B3** (MIT, torch.hub, CPU-viable, el que menos degrada far-field en Interspeech 2025). Ambos sacan speaker ID de la GPU.
6. **Usuario vs TV**: reencuadrar como problema cerrado (Personal VAD: ¿es uno de mis 2-5 enrollados?) + señal contextual: DoA del XVF3800 hacia posición conocida de la TV + estado `media_player` en HA. Coincide con el hallazgo de que el discriminante de los fantasmas es contextual (entidad nombrada + grammar path), no acústico.

### F4 — TTS: spike Pocket TTS (Kyutai) en CPU

- Candidato: Pocket TTS (100M, español desde 2026-05, CC-BY-4.0, streaming real, cloning zero-shot). Runtime C++ INT8 reporta ~30ms TTFA en un Ryzen 3800X; el Threadripper sobra.
- Spike: correr bundles `spanish` (6 capas) y `spanish_24l` (24 capas) en CPU, medir TTFA con las frases cortas de domótica del response cache, A/B de calidad contra Kokoro `ef_dora` (que tiene acento inglés reportado). Probar cloning con muestra rioplatense de 10s.
- Si GO: integrar como engine en DualTTS (la abstracción ya existe — hoy es "Kokoro only") ⇒ **libera ~1-2GB de cuda:1** y saca el TTS de la GPU.
- Plan B si la calidad no convence: Qwen3-TTS-0.6B ya está integrado y deshabilitado en config (esperaba 3ra GPU) — se reactiva cuando lleguen GPUs.

### F5 — Negación: encoder discriminativo (mediano plazo)

- El fallo "no hace falta prender la luz" rompe igual a modelo y gramática (NO-GO del clima). La única vía con evidencia que lo ataca estructuralmente: encoder chico fine-tuneado (**mmBERT-small ~140M** o **EuroBERT-210m**) para intent+slots+clase `no_action` — la negación pasa de instruction following a clase supervisada con ejemplos etiquetados.
- <10ms en GPU, ~20-40ms en CPU. Arquitectura resultante: grammar (conf≥0.75) → encoder → LLM 4B solo para baja confianza.
- Prerequisito: dataset etiquetado con negación bien representada + clase `other/reject` calibrada. El dataset nocturno existente es la base; el held-out del clima es el test.
- No reemplaza al LLM: es un piso nuevo. El encoder no generaliza a intents que no vio.

### F6 — Wake word: shadow LiveKit (baja prioridad)

- El wake actual NO está sordo — la vara es alta, esto es opcional.
- `livekit-wakeword` (abr 2026, Apache 2.0, local): primera mejora arquitectónica real sobre openWakeWord (~100x menos FP/h en su benchmark inglés), entrenamiento en un comando con `target_fp_per_hour` como objetivo.
- Si se hace: entrenar "nexa" (español, `n_samples` 50k, 50-100 voice prompts), correr en **shadow mode** contra el detector actual, criterio go/no-go definido antes. Caveats: path español inmaduro, "nexa" con 2 sílabas es corto para el estándar del área.

---

## Descartes confirmados (no volver a evaluar sin novedad externa)

- **Embeddings**: ningún modelo 2025-2026 arregla antónimos (límite estructural). BGE-M3 se queda; formalizar como invariante que **el verbo nunca entra al embedding** (la polaridad la decide gramática/encoder). Reevaluar solo ante problema de recall real.
- **Speculative decoding en el router**: outputs <20 tokens ⇒ overhead domina; mezcla mal con grammar constraints. Sí probar n-gram self-speculation en el :8200 (respuestas largas, gratis).
- **Sucesores de Whisper**: no existen (turbo sigue siendo el de 2024).
- Cohere Transcribe (alucina en silencio), Kyutai/Moonshine STT (sin español), Picovoice (activación online + 3 usuarios/mes), microWakeWord (para ESP32), ECAPA2 (licencia NC), Phi-4-mini (español flojo), Orpheus/Fish/F5/Zonos/XTTS (VRAM o licencia o sin streaming).

## Watch (revisar al conectar GPUs nuevas)

- **Voxtral Mini 4B Realtime** (mejor STT streaming español open, 16GB) — candidato #1 para GPU nueva.
- **Qwen3-ASR-0.6B** (robustez a ruido como objetivo de RL explícito) — bench contra Parakeet v3 en buckets de vad bajo; cabe en presupuesto actual si algún experimento lo pide antes.
- **Qwen3-TTS 0.6B** (ya integrado, deshabilitado) y **Qwen3.5-4B/9B** (router/slow path cuando madure GGUF de la arquitectura híbrida).
- **Acoustic maps multi-canal** (Neri & Virtanen, EUSIPCO 2026) para humano-vs-parlante con el array — sin código publicado aún; es exactamente el problema TV con exactamente el hardware XVF3800.

## Orden propuesto y dependencias

```
F0.1 (logs shadow) ──→ alimenta Decisión 0
F0.2 (llama.cpp)   ──→ independiente
F0.3 (spike boosting) → independiente
F1 (Qwen3-4B A/B)  ──→ usa held-out clima; independiente de F0
F2 (keep_audio)    ──→ desbloquea F2.1 (LoRA), F3.2 (cohorte), F2.3 (evals)
F3 (speaker ID)    ──→ pasos 1,3 ya posibles; paso 2 espera F2
F4 (Pocket TTS)    ──→ spike independiente; integración toca DualTTS
F5 (encoder)       ──→ espera dataset etiquetado; diseño puede arrancar
F6 (wake LiveKit)  ──→ opcional, al final
```

**Balance de VRAM cuda:1 si F1+F4 dan GO**: ~7.4GB → ~3.5-4GB usados. Ese margen es el habilitador de todo lo demás (y del día que se caiga una GPU). No se reasigna sin discutir.
