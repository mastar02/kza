# Diagnóstico del wake "Nexa" + Roadmap a "Alexa a tope" v2 (2026-05-31)

> Sucesor de `2026-05-31_DIAGNOSTICO_ALUCINACIONES_Y_ROADMAP_ALEXA.md`. Consolida un
> workflow de 10 agentes (research Sonnet + análisis de código Opus + verificación
> adversarial) **corregido con evidencia dura read-only del server**.
>
> ⚠️ **CORRECCIÓN IMPORTANTE:** el informe del workflow afirmó que el wake disparaba
> 1/7 por "un bug del grafo crudo en detector.py". Resultó **FALSO en producción**:
> se diagnosticó sobre el código LOCAL drifteado. La verificación en el server lo
> desmintió. Lección: para diagnosticar prod, ir al server PRIMERO y comparar hashes
> ANTES (ver `feedback_drift_compare_hashes`).

---

## 0. QUÉ PASA REALMENTE CON EL WAKE (verificado en el server, read-only)

**El wake openWakeWord "Nexa" FUNCIONA en producción. No está roto.**

- `detector.py` del server YA enruta `.onnx` por la vía correcta de openWakeWord
  (`elif model.endswith(".onnx"): pretrained.append(model)  # → openwakeword.Model`).
  El "bug del grafo crudo" (`_predict_custom` con MelSpectrogram `[1,40,T]` cuando el
  modelo espera embeddings `[1,16,96]`) **solo existe en la copia LOCAL vieja**.
- **Evidencia en vivo** (run `engine=openwakeword`, PID 3716370, desde 17:34):
  `Wake word in escritorio (nexa: 0.84 / 0.76 / 0.67 / 0.55 / 0.43 …)`; en silencio
  `nexa:0.000` (cero FP); comandos a HA `[HA-CALL] light.turn_on@light.escritorio_2
  success=True`.
- A/B (CPU, read-only) sobre tu voz real: **75% holdout / 93% train**, silencio 0.002.
- Los **16.077 "gracias." de hoy NO son del wake actual** — son de tramos previos con
  `engine=whisper` (PID 1152276, `whisper_wake 🎤`). El run openWakeWord no alucina.
- El stream al detector es **continuo** (contador interno parejo cada 80 ms) → la
  "causa #2" del informe (gates del echo_suppressor cortando chunks) NO domina.

### ¿Por qué "1 de 7"? (diagnóstico honesto)

El wake no está roto, pero la experiencia de "1/7" es real:

1. **Recall far-field genuinamente limitado.** 75% holdout (condiciones controladas)
   baja a distancia/ángulo/ruido. Cerca dispara; lejos falla.
2. **Threshold 0.3 probablemente alto.** Reales puntúan 0.32-0.85, silencio <0.05 →
   margen para bajarlo.

**No se puede fijar el número exacto desde logs pasivos.** Hace falta una **prueba
controlada** (decir "Nexa" N veces a 1/3/5m mirando `[oww-dbg]`, contar recall a
varios thresholds). Palancas: (1) bajar threshold 0.3→0.25/0.2 (gratis, reversible);
(2) más grabaciones reales far-field + reentrenar; (3) earcon de feedback.

**Drift:** `detector.py`, `multi_room_audio_loop.py` y `settings.yaml` local ≠ server.
El server lidera. Comparar `git hash-object` antes de tocar.

---

## 1. Por qué alucina la transcripción (causa raíz — confirmado)

- El **AGC del XVF3800** (`PP_AGCMAXGAIN=64`) amplifica el silencio/ruido por encima
  del gate RMS y del umbral del Silero VAD (apagado). Whisper recibe "ruido con forma
  de voz".
- `faster-whisper-large-v3-turbo` con `vad_filter=False` sobre no-voz regurgita su
  sesgo (subtítulos YouTube): `gracias.`, `gracias por ver el video.`, `¡gracias!`,
  `¡suscríbete al canal!`. (Esto aplica al engine **whisper**; el wake openWakeWord
  actual NO corre Whisper sobre todo el audio, por eso no aluciona en el WAKE — pero el
  **comando** post-wake sí pasa por Whisper y puede alucinar sobre silencio si la
  captura graba una pausa.)
- **Señales de confianza inútiles** (turbo = 4 decoder layers): `no_speech_prob=0.00`
  siempre, `avg_logprob` invertido → **Fase 3 confidence-gate MUERTA**. Confirmado por
  arXiv 2501.11378 y 2505.12969 (Calm-Whisper).
- Cura: no correr Whisper sobre no-voz → VAD real + bajar AGC + ring-prebuffer en la
  captura del comando.

---

## 2. Motor de wake-word: quedarse con openWakeWord

**El modelo "Nexa" funciona (75% holdout, 0 FP).** No cambiar de motor; mejorar recall
(threshold + más reales). **Porcupine DESCARTADO:** Picovoice elimina el free tier el
30/06/2026, los `.ppn` caducan a 30 días y el AccessKey valida contra internet → rompe
el requisito 100% on-prem. `porcupine_wake.py` queda cableado pero inerte.

---

## 3. Explotar el hardware que YA tenés

### 3a. XVF3800 (control por pyusb, VID:PID 2886:001a — verificado en lsusb Bus003 Dev005)
**NO hay tooling instalado** (ni `xvf_host`/`vfctrl`/`dfu-util`/`pyusb`, ni udev rule).
- **(P0, ~15 min) Bajar `PP_AGCMAXGAIN` 64→8** vía `xvf_host` → corta la amplificación
  del silencio → reduce la alucinación del COMANDO Whisper + re-habilita el VAD. A/B con
  voz real (si la voz lejana queda baja, subir a 12-16). **NO** `save_configuration`
  (riesgo brick, issue #8).
- **(P1) Pre-gate `speech energy`/`DOA.speech_active` por pyusb** antes de Whisper
  (0-GPU). **INCIERTO si es pre o post AGC** — validar midiendo. Crear
  `src/audio/xvf_controller.py`.
- **(P1) DOA anti-TV** (suprimir wakes desde el sector de la TV, ±15°).
- **(P2) Canal ch1 (ASR beam)** para Whisper vs ch0 (Conference). Reversible.
- **(P3) AEC loopback** (TTS → playback USB del XVF3800) → barge-in + mata el eco propio.

### 3b. Dongles BLE (UGREEN BT 5.3)
**Límite duro:** los USB no dan timestamps por-paquete → NO room-level (Bermuda lo
documenta). Sirven para **ocupado/vacío binario**.
- **(impacto MUY ALTO / esfuerzo BAJO / costo 0) Pausar STT en cuarto vacío:**
  `PresenceDetector` ya está wired en `main.py` (`get_zone_occupancy()`) pero **nada lo
  consulta** en el pipeline. ~20-30 líneas → elimina alucinaciones del cuarto vacío. No
  necesita timestamps.
- Prior bayesiano BLE→ECAPA (1 persona → bajar umbral 0.75→0.55).
- Room-level real (desambiguación, follow-me) → requiere 2× ESP32 XIAO C6 como
  `bluetooth_proxy` (~10 USD). MAC randomization → IRK enrollment o tag nRF52.

---

## 4. Cómputo y latencia (total ~1001ms; objetivo <300)

- **LLM router Qwen-7B Q4 = 799ms (80% del total).** Expandir la gramática
  determinística (`command_grammar.py`) a climate/volume/alarm/timer → fast path <200ms.
- **Mover Whisper STT a CPU** (Threadripper ocioso) → libera ~1.5GB cuda:1 → mata el OOM.
- **Emotion (185ms) fuera de la ruta crítica** (`asyncio.create_task`).
- SpeakerID lazy. Evaluar Moonshine-base-es solo con A/B de WER rioplatense.

---

## 5. UX clase-Alexa

- **(P0) Earcon <200ms desde el wake:** feedback inmediato → sabés si disparó y
  reintentás. Máximo impacto percibido. (También ayuda con el recall percibido del wake.)
- **(P0) Ring prebuffer ~500ms:** la captura del comando arranca tras el wake; sin
  prebuffer graba silencio AGC → el comando alucina. `deque(maxlen≈10)` siempre lleno.
- **(P0) Endpointing 500→200ms adaptativo** (Silero VAD en CPU, <1ms/frame).
- **(P4) Barge-in:** requiere el AEC loopback de 3a.

---

## 6. Roadmap priorizado

### ESTA SEMANA
1. ✅ **Verificar server** (engine=openwakeword, model=nexa.onnx, thr=0.3; wake FUNCIONA) — HECHO.
2. **Prueba controlada del recall** (decir "Nexa" 15-20× a 1/3/5m, contar % >0.3/0.25/0.2). `[define todo][30min][nulo]`
3. **Bajar threshold** según (2) — 0.3→0.25/0.2 si el silencio se mantiene <0.05. Reversible. `[alto][trivial][bajo]`
4. **Bajar `PP_AGCMAXGAIN` 64→8** (instalar tooling XVF3800). `[alto][15min][bajo]`
5. **Earcon <200ms.** `[muy alto percibido][2h]`
6. **Pausar STT en cuarto vacío** (hook BLE). `[muy alto][bajo][bajo]`
7. **Ring prebuffer 500ms** + **endpointing 500→200ms.** `[alto][medio]`

### DESPUÉS
- Pre-gate spenergy pyusb → DOA anti-TV. Mover STT a CPU. Emotion async.
- Expandir gramática (climate/volume/alarm/timer). Canal ch1 ASR. AEC loopback → barge-in.
- Más "Nexa" reales far-field + reentrenar (si el recall no sube con threshold).
- 2× ESP32 C6 + Bermuda + IRK (room-level).
- (opcional, NO urgente) sync del detector.py local→server + quitar `[oww-dbg]`.

---

## 7. Métricas de éxito

| Métrica | Estado | Objetivo |
|---|---|---|
| Wake recall far-field | ~? (medir en vivo; holdout 75%) | >85% |
| Wake FP sobre silencio (oWW) | 0 (nexa:0.000) ✓ | <0.5/h |
| Alucinación del COMANDO Whisper | presente (sin prebuffer + AGC alto) | ~0 |
| Latencia voz→acción p50 | ~1001ms | <300ms |
| Earcon desde wake | ninguno | <200ms |
| VRAM libre cuda:1 | ~0.8-1.5GB (OOM) | >2.5GB (STT→CPU) |

---

## 8. Decisiones abiertas

1. ¿Hacemos la prueba controlada de recall del wake ahora (vos decís "Nexa", yo leo `[oww-dbg]`)?
2. ¿Bajar threshold 0.3→0.25/0.2 tras medir? (reversible, en `settings.yaml` server).
3. ¿Instalar el tooling del XVF3800 y bajar AGC 64→8 esta semana?
4. ¿Cuál es el próximo frente: alucinaciones (AGC), UX (earcon/prebuffer), BLE (pausar vacíos) o latencia (router/STT→CPU)?

---

## Fuentes
arXiv 2501.11378, arXiv 2505.12969 (Calm-Whisper), faster-whisper, openWakeWord
(dscripka/CoreWorxLab), Picovoice docs (free tier EOL 30/06/2026), Bermuda,
HA Private BLE Device, Seeed XVF3800 wiki + XMOS user guide, AVS pre-roll 500ms.
Workflow run `wf_3963dee4-62c` (corregido con verificación server-side).
