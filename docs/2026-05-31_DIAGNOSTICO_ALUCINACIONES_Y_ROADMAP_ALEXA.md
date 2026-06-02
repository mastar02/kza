# Diagnóstico de alucinaciones + Roadmap a "Alexa a tope" — 2026-05-31

> Consolida evidencia dura del sistema en producción (76.754 wake events, 1.070 requests, 2026-04-24..05-31) + investigación de 7 agentes (XVF3800, Whisper, wake-word engines, BLE, UX far-field, código, cómputo) con fuentes citadas. Trabajo de campo en el server `kza@192.168.1.2`.

---

## TL;DR

1. **Por qué alucina:** KZA corre Whisper sobre **cada ventana de audio sin VAD efectivo**. El AGC del XVF3800 (`PP_AGCMAXGAIN=64`) amplifica el silencio/ruido por encima del gate RMS (`min_rms=0.025`), el Silero VAD está **apagado** (`use_silero_vad=false`, porque el AGC lo rompía), y `vad_filter=False` en la llamada a Whisper. Resultado: Whisper transcribe ruido amplificado y emite el sesgo de su training (subtítulos de YouTube): `"¡Gracias!"` ×14.224, `"Gracias por ver el video."` ×8.358. Las señales de confianza de Whisper **no pueden atraparlo** (`no_speech_prob=0.00` siempre; `avg_logprob` invertido: alucinaciones −0.42 vs comandos reales −0.81).
2. **Por qué estamos lejos de Alexa:** es **arquitectónico**. KZA usa "Whisper-como-wake-word" (un *stopgap* explícito en el docstring), no un wake-word dedicado siempre-on. La arquitectura correcta (openWakeWord) **ya está codeada y es el default** — sólo está bypasseada porque nunca se entrenó un modelo "Nexa". Además el XVF3800 es un front-end far-field clase-Echo cuyas capacidades (VAD/DOA on-chip por USB, AEC con loopback, selección de beam) **no se usan**.
3. **El camino:** (F1) frenar el flood ya, (F2) flipear a wake-word dedicado, (F3) explotar el XVF3800, (F4) bajar latencia, (F5) presencia/UX. Detalle abajo.

---

## Parte A — Evidencia medida (sistema en producción)

| Métrica | Valor | Fuente |
|---|---|---|
| Wake events totales | 75.696 | `kza-metrics.jsonl` |
| Tasa de aceptación | **1.4%** (1.087) | idem |
| Aceptados desde instrumentación (30-05) | **8** vs 14.285 rechazos | idem |
| `total_ms` request (median / p90) | **1.001 ms / 5.788 ms** (objetivo <300) | 1.070 requests |
| `llm_router_ms` (cuando se usa) | **median 799 ms** | 332 requests |
| `emotion_ms` (inline) | median 185 ms | 237 requests (¡activo en server!) |
| `vector_search_ms` | median 16 ms, p90 203, max 1.706 | 821 requests |
| `home_assistant_ms` | median 0, p90 57 | 821 requests |

**Top textos alucinados (rechazados):** `¡Gracias!` ×14.224 · `Gracias por ver el video.` ×8.358 · `Gracias.` ×6.910 · `¡Suscríbete al canal!` ×2.181 · `Amén.` ×222 · `¿Verdad?` ×200 · `Adiós.` ×134 · `Esto es un asistente de voz de la luz del escritor…` ×128 (← **el `initial_prompt` regurgitado**).

**Reject reasons:** `below_fuzzy_threshold` 44.020 · `no_speech_hallucination` 12.722 · `tv_stop_phrase` 7.264 · `no_keyword` 4.975 · `no_command_verb` 2.202 · `multi_wake_hallucination` 1.774 · `implausible_speech_rate` 729 · `pathological_repeat` 623.

**Patrón temporal:** alucinaciones 24/7, **~500/hora a las 4-7am** (silencio, TV apagada) → confirma que es Whisper-sobre-silencio, NO audio de TV.

**Señales de confianza (14.288 eventos instrumentados):**
- `no_speech_prob = 0.00` en TODOS (aceptados y rechazados) → inservible.
- `avg_logprob` **invertido**: alucinaciones median −0.42 (mejor), comandos reales median −0.81 (peor). Imposible separar por umbral.

**Config deployada relevante** (`settings.yaml` server):
- `wake_word: {model: hey_jarvis, inference_framework: onnx, threshold: 0.35}` ← scaffolding openWakeWord presente, modelo genérico.
- `stt: {model: whisper-v3-turbo, device: cuda:1, compute_type: int8_float16, beam_size: 5, vad_filter: True, initial_prompt: "Esto es un asistente de voz llamado Nexa…"}`.
- Wake detector (`whisper_wake.py:827-828`): `vad_filter=False, condition_on_previous_text=False`.
- `command_gate: {enforce_confidence: False, max_no_speech_prob: 0.6, min_avg_logprob: -1.2}` ← shadow mode; umbrales inútiles dado lo anterior.

**Hardware confirmado:** `lsusb` → `2886:001a Seeed reSpeaker XVF3800 4-Mic Array`; ALSA card 1, USB `3-1.4`, 2 canales UAC (FL/FR, S16_LE, 16 kHz) + 2 canales playback. **No hay tooling de control instalado** (ni `xvf_host`/`vfctrl`/`dfu-util`/pyusb).

---

## Parte B — Causa raíz mecanística (cadena completa)

```
TV / ruido ambiente / SILENCIO
      │
      ▼
XVF3800 DSP: AEC + beamforming + noise-suppression + AGC (PP_AGCMAXGAIN=64)
      │  el AGC NORMALIZA el nivel de salida → el "silencio" sale amplificado a nivel "voz"
      ▼
Gate RMS (min_rms=0.025)  ──►  el ruido amplificado SUPERA el umbral → pasa
      │  (Silero VAD apagado: use_silero_vad=false, porque el AGC lo hacía devolver prob≈0)
      ▼
faster-whisper.transcribe(vad_filter=False, condition_on_previous_text=False)
      │  Whisper-large-v3-TURBO (4 decoder layers vs 32 → MÁS propenso a alucinar)
      ▼
Decoder autoregresivo sin señal acústica de voz → emite el sesgo del training
   = "Gracias por ver el video", "¡Suscríbete!", "¡Gracias!"  (+ regurgita el initial_prompt)
      │
      ▼
Post-proceso: estas frases NO tienen wake+verbo → rechazadas, PERO:
   • desperdician ~20% de GPU corriendo Whisper sobre nada
   • alimentaban TV-mode espurio (ya parcheado)
   • un subset CON wake+verbo alucinado SÍ se acepta → acción fantasma
```

**Confirmación de literatura:**
- [arXiv:2501.11378](https://arxiv.org/html/2501.11378v1): sin VAD **40.3%** de alucinación sobre no-speech → con Silero VAD pre-filter **0.2%** (reducción **200×**). Frase #1 alucinada: "thank you" (= nuestro "¡Gracias!"). `no_speech_prob` no es predictor fiable.
- [arXiv:2505.12969 Calm-Whisper](https://arxiv.org/html/2505.12969v1): 3 attention heads del decoder causan >75% de las alucinaciones; en turbo (4 layers) su peso relativo es mayor → turbo alucina más.
- Por qué `avg_logprob` invertido: "Gracias por ver el video" es una secuencia de 5-7 tokens ultra-frecuente = mínimo de energía profundo = alta confianza; un comando rioplatense ("Nexa bajá la luz") tiene tokens raros (nombre propio, voseo) = más incertidumbre.

**Conclusión de debugging (Phase 1 cerrada):** la Fase 3 del plan previo (gate por confianza acústica) está **muerta** — ninguna señal de Whisper discrimina. El fix robusto es **no correr Whisper sobre no-voz**: VAD real + wake-word dedicado + (opcional) VAD/DOA on-chip del XVF3800.

---

## Parte C — Inventario de hardware: lo que tenemos y NO usamos

### XVF3800 (Seeed 4-Mic Array, XMOS) — front-end clase-Echo subutilizado
Fuentes: [XMOS User Guide v3.2.1](https://www.xmos.com/documentation/XM-014888-PC/pdf/xvf3800_user_guide_v3.2.1.pdf), [Seeed Wiki](https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/), [repo respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY](https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY).

| Capacidad | Detalle | Estado en KZA |
|---|---|---|
| 4 mics MEMS, beamforming adaptativo (3 beams + auto-select) | Refuerza la dirección del hablante, atenúa fuentes puntuales (TV) | Parcial (usamos el canal procesado) |
| Noise suppression DNN hasta 25 dB + dereverb | On-chip, 0 CPU host | Implícito (viene en el canal) |
| **VAD on-chip** (`DOA_VALUE.speech_active`, 0/1) | Lectura por **pyusb** (USB vendor control), <1 ms, 0 GPU, opera **pre-AGC** | ❌ **NO usado** |
| **DOA 0-359°** (`AEC_AZIMUTH_VALUES`/`DOA_VALUE`) | Ángulo de arribo → filtrado espacial anti-TV | ❌ NO usado |
| **AEC con loopback** | Requiere inyectar el audio del TTS al canal izq. del playback USB | ❌ probablemente sin referencia (TTS → MA1260, no vuelve al chip) → AEC degradado |
| Canal 0 "Conference" vs canal 1 "ASR" (beam auto-select) | El ch1 es el optimizado para ASR | ⚠️ verificar cuál lee KZA |
| Firmware 6 canales (4 raw + 2 procesados) | Permite Silero VAD sobre raw + Whisper sobre beam | ❌ no flasheado |
| `PP_AGCMAXGAIN` (default 64) | Bajarlo a 8-16 re-habilita Silero VAD | ❌ default |
| Wake-word en firmware | **NO existe** — hay que correr openWakeWord en host | n/a |
| Cert. | Microsoft Teams (NO Amazon AVS) | tuning de conferencia, puede requerir ajuste |

**Acceso en Linux:** UAC2.0 nativo (sin drivers). Control vía `pyusb` (puro Python, OK en 3.13) + udev rule `2886:001a MODE=0666`. El repo de Seeed trae `xvf_host` (binario linux_x86_64) y `python_control/xvf_host.py`.

### Cómputo (2× RTX 3070 8GB + Threadripper 7965WX 24c/48t)
- La GPU que hostea STT (`cuda:1` según settings.yaml actual) está **al límite**: Whisper turbo int8 (~1.5 GB) + ECAPA SpeakerID (~400 MB) + Kokoro TTS (300 MB modelo / **1-2.5 GB en inferencia**, buffers cuBLAS) + Qwen-7B Q4 fast-router (~4-5 GB) ≈ **7-9 GB** → de ahí los OOM al reiniciar.
- Threadripper 24c/48t **ocioso**: ideal para mover a CPU el wake-word (openWakeWord ~0.5% de un core), VAD (Silero/WebRTC <1 ms), embeddings (BGE-M3 ya en CPU, 48 ms p50), y ECAPA SpeakerID (~10-30 ms).
- Latencia: el cuello NO es el STT (reusa transcripción del wake, ~0 ms). Es (a) `silence_end_ms=500` del wake + (b) `llm_router` ~800 ms cuando se usa (modelo débil Q4) + (c) `emotion` 185 ms inline + (d) spikes de vector search.

### BLE (dongles USB UGREEN BT 5.3, uno por habitación)
Fuentes: [Bermuda](https://github.com/agittins/bermuda), [HA Private BLE Device](https://www.home-assistant.io/integrations/ibeacon/), [ScienceDirect 2024 RSSI re-id](https://www.sciencedirect.com/science/article/pii/S0167404824003857).

- **Limitación física clave:** un host triangula por cuarto **sólo si cada dongle está físicamente en un cuarto distinto** (KZA los tiene "por habitación" → OK para presencia "nearest-anchor"). Pero el backend BT de HA **no expone timestamps por paquete para dongles USB** (sí para proxies ESPHome) → calidad/latencia degradada (~10 s vs ~2 s).
- **MAC randomization** (iOS/Android) rompe el tracking por MAC → usar **IRK enrollment** (HA Private BLE Device) o **BLE tags nRF52 con MAC estática** (~$5-8) como "ancla de persona". AirTag/Tile **no sirven** (protocolo cerrado).
- **Qué aporta a la voz** (como **prior/contexto**, NO como gate real-time — latencia 5-30 s):
  1. **Pausar el pipeline Whisper de cuartos vacíos** → elimina ~500 alucinaciones/h de ese mic. ⬅ impacto directo en el bug.
  2. **Prior bayesiano para speaker-ID (ECAPA)**: si BLE confirma 1 sola persona en el cuarto, ECAPA baja su umbral 0.75→~0.55 → más identificaciones correctas.
  3. Desambiguación de comandos ("prendé la luz" → cuarto ocupado) — refuerza lo que ya hace `room_context`.
  4. Routing de TTS / follow-me audio al MA1260.
- **Mejora opcional ($10):** 2× ESP32 XIAO C6 con `bluetooth_proxy` → timestamps + latencia ~2 s.

---

## Parte D — Brecha vs Alexa (qué falta para que "se sienta" Alexa)

| Pilar Alexa | Estado KZA | Gap |
|---|---|---|
| Wake reliability (≈0 FA, ≈0 miss) | 1.4% accept, flood de FA | **Crítico** → wake dedicado |
| Far-field robustness | XVF3800 lo da, mal aprovechado | Medio → usar beam/AEC |
| Latencia percibida (<1 s, earcon inmediato) | ~1 s median, sin earcon | Alto → earcon + endpointing |
| Barge-in (interrumpir al TTS) | No | Medio → AEC loopback + double-talk |
| Endpointing (sabe cuándo terminaste) | `silence_end_ms=500` fijo | Medio → VAD endpointing |
| Feedback no-verbal (LED/tono al wake) | No | Alto-percepción → earcon + LED ring |
| Multi-turn / contexto | Parcial | Medio |
| TTS natural | Kokoro OK | Bajo |

**Rúbrica medible para trackear progreso:** wake FA/hora, wake miss-rate, latencia p50/p95 voz→acción, intent-success-rate, barge-in sí/no.

---

## Parte E — Roadmap priorizado

> Notación: **[I]** impacto, **[E]** esfuerzo, **[R]** riesgo.

### Fase 1 — Frenar el flood YA (días, riesgo bajo, sin hardware nuevo)
1. **Activar `vad_filter=True` en el wake detector** (`whisper_wake.py`), params permisivos (`min_silence_duration_ms≈150, speech_pad_ms≈80, threshold≈0.35`). **[I] alto · [E] 1h · [R] bajo** (subir `pre_roll_ms` 200→300 para no comer la 1ª sílaba).
2. **Acortar el `initial_prompt`** del STT/wake (hoy se regurgita ×128) a algo mínimo o vacío + `hotwords="Nexa"`. **[I] medio · [E] 30m**.
3. **`beam_size=1`** en el path de wake (menos alucinación + menos latencia). **[I] medio · [E] 15m**.
4. **Denylist de utterance-completa ampliada** desde el top del log (`gracias`, `de nada`, `buenas noches`, …) — red barata, ya parcialmente implementada. **[I] medio · [E] 1h**.
5. **NO** activar `enforce_confidence` por confianza acústica: está demostrado inútil (señales no discriminan). Mantener gate por reglas de texto.

### Fase 2 — Arquitectura correcta: wake-word dedicado (la palanca grande)
> Elimina el flood **en origen** y baja la latencia de wake de ~580 ms a <150 ms. La infra ya existe (`WakeWordDetector` + `WakeWordTrainer` + selector en `main.py:668`).

6. **Entrenar un modelo "Nexa"** (openWakeWord, salida **`.onnx`** — obligatorio en Python 3.13, tflite no tiene wheels). Pipeline [CoreWorxLab/openwakeword-training](https://github.com/CoreWorxLab/openwakeword-training) con Kokoro (ya instalado) + **20-50 grabaciones reales** ponderadas 3× (el español en openWakeWord es flojo sin voz real: recall ~0.41 sólo-sintético). **[I] máximo · [E] 1-2 días · [R] medio (calidad ES)**.
7. **Flipear el engine** del cuarto a `openwakeword` y conectar: wake (CPU, siempre-on) → ring prebuffer (~0.8 s) → **Silero VAD para endpointing** (ahora corre sólo sobre el clip corto → vuelve a ser compatible con el XVF3800) → Whisper STT sólo sobre el clip. **[E] 2-3 días**.
8. **Fallback Porcupine** si el recall de "Nexa" en ES no alcanza ≥70%: entrena desde texto en segundos, free para uso personal, OK en 3.13 (riesgo: vendor lock-in). **[E] horas**.
9. **Cleanup:** remover el loop "Whisper-como-wake", `StreamingSTT` (código muerto), consolidar las dos implementaciones de VAD.

### Fase 3 — Explotar el XVF3800 (hardware ya instalado)
10. **Instalar el control USB** (`pyusb` + udev `2886:001a`) y leer `DOA_VALUE` → pre-gate `speech_active` 0-GPU antes del wake/Whisper (defensa extra, opera pre-AGC). **[I] alto · [E] 2-4h**.
11. **Bajar `PP_AGCMAXGAIN` 64→8-16** y verificar que Silero VAD vuelve a discriminar (re-habilita una 2ª capa barata). **[E] 1h en server**.
12. **Verificar el canal** que lee KZA (ch0 "Conference" vs ch1 "ASR") y usar el mejor; abrir el stream con los canales nativos (ya se hace 2ch). **[E] 30m**.
13. **AEC con loopback:** rutear el TTS también al playback USB del XVF3800 (canal izq.) → cancela el eco del propio asistente → habilita barge-in. **[I] medio-alto · [E] 4-8h + tuning `AUDIO_MGR_SYS_DELAY`**.
14. (Opcional) **DOA anti-TV:** calibrar la dirección de la TV por cuarto y rechazar wakes desde ese sector. **[E] 3-5h**.

### Fase 4 — Latencia a <300 ms percibida + UX clase-Alexa
15. **Earcon/tono + LED al detectar wake** (feedback inmediato) — cambia la percepción aunque el backend tarde igual. **[I] alto-percepción · [E] 2-4h**.
16. **Endpointing por VAD** (silencio adaptativo ~400 ms) en vez de `silence_end_ms=500` fijo.
17. **Sacar `emotion` (185 ms) del path crítico** (correr async/post-acción) y **mover ECAPA/embeddings a CPU** (libera VRAM + saca de la ruta caliente).
18. **Atacar `llm_router` (~800 ms):** el grammar fast-path ya evita el LLMRouter Q4 débil para domótica; ampliar su cobertura y/o subir el quant del router para queries conversacionales.

### Fase 5 — Presencia BLE como contexto
19. **Bermuda (HACS) + asignar cada dongle a un Area** + IRK enrollment / BLE tags nRF52. **[E] 2-4h**.
20. **Pausar pipeline en cuarto vacío** (consume estado HA) → menos alucinaciones + menos cómputo.
21. **Prior bayesiano BLE→ECAPA** en `speaker_identifier.py`. **[E] 4h**.
22. (Opcional $10) 2× ESP32 XIAO C6 como proxies BT con timestamps.

---

## Decisiones abiertas (requieren input)
- **Motor de wake:** openWakeWord (libre, ES incierto, reentrenable) vs Porcupine (fácil, propietario, lock-in). Recomendación: intentar openWakeWord con voz real; Porcupine como red de seguridad.
- **STT alternativo:** mantener Whisper turbo + VAD (recomendado) vs evaluar NeMo Parakeet (CTC, no alucina por diseño, pero flojo en rioplatense) o Moonshine-ES (libera VRAM, WER ES incierto). Sólo en staging.
- **Secuencia:** ¿parche rápido F1 primero (1 día, baja el flood) y después la re-arquitectura F2? ¿o directo a F2?

## Fuentes
Ver informes completos en `/tmp/kza_reports/*.md` (xvf3800, whisper, wakeword, ble, ux, código, cómputo) — cada uno con su bibliografía citada.
