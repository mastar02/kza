# ReSpeaker XVF3800 en KZA: cómo exprimirlo al máximo, y la comparación honesta con el Flex y el resto del mercado

**Fecha:** 2026-06-03
**Branch analizado:** `feat/nexa-command-detection-fixes`
**Audiencia:** dueño de KZA
**Contexto:** openwakeword "Nexa" → Whisper large-v3-turbo far-field, success ~12%, alucinaciones / AGC inflado / STT garbleado. Pre-gate SPENERGY (commit `28cd4d5`) y pre-roll 1.0s (commit `400942c`) ya en producción.

---

## TL;DR

El XVF3800 que ya tenés es **lo mejor que hay** en su clase consumer/maker para captura far-field. El cuello de botella es **~90% software/firmware, no el micrófono**. Hoy KZA usa **exactamente 1 capacidad del chip** (una lectura read-only de SPENERGY como gate VAD) de un DSP que ofrece beamforming dirigible, AEC con referencia, control de AGC, re-ruteo de canal ASR, DoA y más. Tres quick-wins de muy bajo esfuerzo (cerrar el bypass del gate en early_dispatch, pinear `pyusb`, pasar la config de VAD al loop) protegen y completan trabajo ya hecho. El lever grande es convertir el `XvfController` de sensor pasivo a **front-end configurable** (write USB) para bajar el AGC y, posiblemente, capturar el canal ASR — pero ambos hay que **validarlos con datos**, no a ciegas. Migrar a reSpeaker Flex **no mueve la aguja** (mismo chip, mismo firmware).

---

## 1. Qué tenemos y cuánto lo usamos hoy

El XMOS XVF3800 (4 mics MEMS, 360°, far-field ~5m) trae un pipeline de DSP completo: AEC → beamforming (3 beams: 1 scan + 2 focused + auto-select) → dereverb → noise-suppression DNN → EQ → AGC → limiter, más DoA y VAD por hardware. KZA toca el chip de **una sola forma**: una lectura read-only del registro `AEC_SPENERGY_VALUES` (resid 33, cmd 80) vía control-transfer pyusb, usada como pre-gate antes de Whisper (`src/audio/xvf_controller.py:34-37`, `:100-125`).

| Capacidad del XVF3800 | Estado en KZA | Evidencia (file:line / config) |
|---|---|---|
| **SPENERGY** (VAD hardware, pre-AGC, 0-GPU) | ✅ **Usado** — gate binario antes de Whisper, solo índice 3 (auto-select beam) | `src/audio/xvf_controller.py:153,164`; `src/pipeline/multi_room_audio_loop.py:522-544`; `config/settings.yaml:1037-1041` |
| 4 mics MEMS / array | ⚠️ **Parcial** — el stream USB se abre con canales nativos pero downstream se consume **solo `indata[:,0]`** | `src/pipeline/multi_room_audio_loop.py:32-47`, `:383` |
| Canal 0 (Conference) vs Canal 1 (ASR / auto-select beam) | ❌ **Se usa el canal Conference** (post-proceso pesado para oído humano), no el ASR | `src/pipeline/multi_room_audio_loop.py:383` |
| **Escritura de cualquier parámetro** (AGC, beam, routing) | ❌ **Cero writes** — `XvfController` es read-only por diseño | grep confirma 0 `ctrl_transfer 0x40` / CTRL_OUT en todo el repo |
| AGC / MAXGAIN (PP_AGCMAXGAIN=64 medido, infla el piso a ~0.025-0.05) | ❌ **Nunca se baja por código** — solo se documenta como "calibrar a mano" | `multi_room_audio_loop.py:148-154`; `config/settings.yaml:1025-1029` |
| DoA / azimuth | ❌ **No se lee** (descartado como gate único; ver §5) | — |
| Beamforming dirigido / fijar beam | ❌ **No se usa** — solo se lee SPENERGY[3] pasivamente | `xvf_controller.py:37` |
| AEC referenciado al TTS (barge-in hardware) | ❌ **Ciego** — el TTS sale por el MA1260, no vuelve al playback USB del chip; KZA usa AEC en host por correlación numpy | `src/audio/echo_suppressor.py:234,339` |
| VAD on-chip (speech_active) | ❌ **No se lee**; `use_silero_vad:false`, el VAD real es RMS en software | `config/settings.yaml:1073` |
| GPIO / LED ring / mute hardware / canales raw pre-beamforming | ❌ **No se usan** | — |
| Versión firmware / serial vía vendor interface | ❌ **No se lee** (serial hardcodeado en comentario) | `config/settings.yaml:1145` |

**Resumen brutal:** de N capacidades del DSP, KZA aprovecha **exactamente 1** (SPENERGY como gate binario). Además, el `xvf_host` oficial de Seeed **no está vendoreado** (solo se portó la función de lectura de SPENERGY), y **`pyusb` no está pineado** en `requirements.txt` → riesgo de que el gate quede silenciosamente OFF en deploy (ver §2, palanca QW-2).

---

## 2. Palancas para exprimir el XVF3800 (priorizadas)

Marco de verificación: **confirmado** = código + fuente primaria coinciden; **a validar con datos** = el lever es viable y la mecánica está confirmada, pero un número o el efecto concreto no está medido y debe probarse A/B; **advertencia** = corrección a una afirmación errónea de la propia investigación.

### Quick wins (bajo esfuerzo, alto impacto, confirmados)

#### QW-1 — Cerrar el bypass del gate SPENERGY en el path early_dispatch
- **Estado:** ✅ **Confirmado** (doble verdict adversarial).
- **Problema:** el gate SPENERGY solo corre en el fallback VAD-silencio (`multi_room_audio_loop.py:354`). El bloque early_dispatch (`:329-347`) despacha el `CommandEvent` **sin** llamar a `_passes_spenergy_gate`, y con `early_dispatch: true` (`config/settings.yaml:1097`) ese es el path **más rápido y más usado**. Una alucinación con forma de comando (o speech garbleado sobre ruido) que el grammar resuelve rápido se ejecuta **saltándose el VAD por hardware** — justo la señal que mata esas alucinaciones (secador=0, voz≥52k, threshold=100).
- **Cambio concreto:** duplicar el chequeo del gate en el bloque early_dispatch antes de `asyncio.create_task(self._dispatch_command(event))`. **Opción recomendada (b):** replicar las líneas 354-356 dentro de `:329-347` (la var `rs` ya está en scope). **No** uses la opción (a) "mover al inicio de `_dispatch_command()`" tal cual: `CommandEvent` no lleva `command_start_time`, solo un `timestamp`, así que `peak_since(event.timestamp)` daría una ventana casi vacía → gate degradado. Requeriría plumbing extra.
- **Impacto:** alto. **Esfuerzo:** bajo. El gate ya está implementado, calibrado y es fail-open.
- **Matiz honesto:** las alucinaciones clásicas de silencio ("Gracias por ver el video") no parsean a comando full, así que no disparan early_dispatch igual. El valor real es contra alucinaciones/garble *con forma de comando*. Cierra un hueco real a costo casi nulo.

#### QW-2 — Pinear `pyusb` y verificar la udev rule (garantía operativa)
- **Estado:** ✅ **Confirmado** (doble verdict). Riesgo **vivo**, no hipotético: `pip show pyusb` en el venv local da "not found".
- **Problema:** `import usb.core` es lazy y fail-open (`xvf_controller.py:85-88`). Si el venv de prod no tiene pyusb, el pre-gate SPENERGY queda **silenciosamente OFF** con solo un WARNING, devolviendo exactamente las alucinaciones que el gate debía matar. `pyusb` **no aparece** en `requirements.txt` (grep 0 hits; nunca se agregó cuando aterrizó SPENERGY en `28cd4d5`).
- **Cambio concreto:** agregar `pyusb==1.3.1` a `requirements.txt` (release real 2025-01-08, requiere Python≥3.9, compatible con el 3.13 de KZA). Verificar que la udev rule `MODE=0666` (`deploy/udev/99-xvf3800.rules:10`, VID:PID 2886:001a) esté aplicada en el server para acceso non-root del user kza — sin ella, `usb.core.find` falla por permisos y el gate también queda OFF.
- **Impacto:** alto (protege todo el trabajo de SPENERGY). **Esfuerzo:** bajo.

#### QW-3 — Pasar la config de VAD/silencio de settings.yaml al MultiRoomAudioLoop
- **Estado:** ✅ **Confirmado** (verdict directo en código).
- **Problema:** `main.py:838-861` construye el `MultiRoomAudioLoop` **sin** pasar `silence_threshold`, `silence_duration_ms` ni `min_speech_ms`, así que el loop usa hardcoded defaults (`0.015/300/300`, `multi_room_audio_loop.py:95-97`). Los valores calibrados `rooms.wake_word.min_rms:0.007` (`settings.yaml:1071`) y `silence_end_ms:500` (`:1063`) **solo** alimentan al `WhisperWakeDetector`, que es el engine **inactivo** (`engine: openwakeword`, `settings.yaml:1012`). El corte de fin-de-comando del path de producción usa umbrales no afinados para el XVF3800.
- **Cambio concreto:** cablear los tres parámetros por constructor (patrón DI del proyecto). **Cuidado conceptual:** `min_rms` es un gate de *energía de ingreso* (wake-time), `silence_threshold` es un detector de *silencio de fin-de-comando* — no son intercambiables 1:1. No pases `0.007` directo a `silence_threshold` sin pensar; calibrá el endpointing aparte.
- **Impacto:** medio. **Esfuerzo:** bajo.

### Lever grande (esfuerzo medio, alto impacto)

#### L-1 — Extender XvfController para ESCRIBIR parámetros del chip
- **Estado:** ✅ **Confirmado** (protocolo verificado contra el `xvf_host.py` oficial de Seeed, byte por byte).
- **Qué desbloquea:** convierte el board de sensor pasivo a front-end configurable, sin tooling externo y sin tocar el audio streaming (todo va por el Interface 3 vendor).
- **Protocolo EXACTO:** el read actual usa `bmRequestType=0xC0` (CTRL_IN|VENDOR|DEVICE), `wValue=0x80|cmd`, `wIndex=resid` (`xvf_controller.py:33-42,111`). El **write es simétrico**: `bmRequestType=0x40` (CTRL_OUT|VENDOR|DEVICE), `wValue=cmd` **sin** el bit `0x80`, `wIndex=resource_id`, `data=bytes del valor`. Los resid/cmdid salen del dict `PARAMETERS` del `xvf_host.py` oficial (mismo del que se portó SPENERGY = resid 33/cmd 80). Parámetros writables confirmados (`rw`): `PP_AGCMAXGAIN` (17/11), `PP_AGCONOFF`/`PP_AGCGAIN`, `AEC_FIXEDBEAMSONOFF` (33/37), `AEC_FIXEDBEAMSAZIMUTH_VALUES` (33/81), `AEC_FIXEDBEAMSGATING` (33/83), `AUDIO_MGR_OP_L`/`AUDIO_MGR_OP_R` (35/15, 35/19).
- **Diseño:** fail-open igual que el read (si el write falla, dejar el chip en su config guardada). **Nota de cita:** el `xvf_host.py` con el `PARAMETERS` dict vive en `python_control/xvf_host.py` del repo Seeed, no en `host_control/` (ahí solo hay binarios).
- **⚠️ Regla de oro:** escribir **en RAM, NUNCA `save_configuration`**. El issue #8 del repo Seeed documenta que `save_configuration 1` puede dejar el device sin enumerar (solo Safe Mode; el DFU no resetea config). Los writes a RAM son **reversibles desenchufando**. Antes de cualquier cirugía, tener tar backup.
- **Impacto:** alto (habilita L-2, L-3, L-4). **Esfuerzo:** medio (re-portar el command-map + tests con cuidado).
- Fuentes: `github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/blob/master/python_control/xvf_host.py`; `xmos.com/.../03_using_the_host_application.html`.

#### L-2 — Bajar el AGC para que deje de inflar el piso de ruido
- **Estado:** mecanismo **confirmado**, número/efecto exacto **a validar con datos**.
- **Lo confirmado:** el AGC está ACTIVO y amplificando (lectura directa del chip: `PP_AGCONOFF=1`, `PP_AGCGAIN≈8.06`, `PP_AGCMAXGAIN=64`). El `64` es el **preset de Seeed, no el default del chip**. El AGC actúa **post-AEC/beamforming**, así que bajarlo **no afecta** el pre-gate SPENERGY (pre-AGC). El comentario en código ya documenta que el AGC ×64 infla el piso a ~0.025-0.05 y por eso `min_wake_rms` está en `0.0` (off, `settings.yaml:1029`) y Silero está off (`:1073`).
- **Dos vías (elegir UNA, no ambas):**
  - **Vía A — recortar el techo:** `PP_AGCMAXGAIN 64 → 8-16`. Si la voz lejana (3-4m) queda baja, subir a 12-16. Re-habilitaría `min_wake_rms` y potencialmente Silero VAD.
  - **Vía B — congelar el gain (modo ASR-ish):** `PP_AGCONOFF=0` + `PP_AGCGAIN` fijo moderado calibrado a voz a 3-4m. Mata el "pumping" del piso en silencios.
- **⚠️ Advertencias sobre la justificación (corregir antes de citar):**
  - El "default de fábrica XMOS ~31.6/30dB" que circuló es **falso**: el default real de `PP_AGCMAXGAIN` es **125.0** (lineal) y el de `PP_AGCGAIN` es **32.0**. O sea, Seeed en realidad **bajó** el techo respecto del chip (125→64). El argumento direccional ("64 ≠ default del chip") es correcto, el número con que se justificaba no.
  - El "modo ASR con ganancia fija" que documenta XMOS **no es** `PP_AGCONOFF=0`: es un path/parámetro separado (`AEC_ASROUTGAIN`, tapeado del beamformer **antes** del post-processor, sin NS). Congelar el AGC es una vía *alternativa* legítima al mismo objetivo, pero no es literalmente lo que la doc llama "modo ASR".
  - La cadena causal "AGC bombea el piso → infla RMS → dispara alucinaciones" es **inferencia plausible, no medición**. **Validar:** medir RMS del piso con AGC alto vs bajo, y success rate antes/después.
- **Impacto:** alto. **Esfuerzo:** medio (requiere L-1). Ataca el problema que SPENERGY **no** resuelve: STT garbleado + ruido amplificado.

#### L-3 — Probar el canal 1 (ASR) en vez del canal 0 (Conference)
- **Estado:** ⚠️ **A validar con datos** (un verdict lo dio confirmado, otro incierto — la asignación L/R del firmware flasheado NO está verificada en KZA).
- **Lo confirmado:** hoy `multi_room_audio_loop.py:383` toma **siempre `indata[:,0]`**, y la doc Seeed/XMOS describe canal izquierdo = Conference (post-proceso pesado para oído humano) y canal derecho = ASR (beam auto-select, menos destructivo). El stream del XVF3800 ya se abre con 2ch nativos, así que `indata[:,1]` es accesible.
- **Lo incierto:** los **propios docs de KZA** marcan "verificar cuál lee KZA (ch0 Conference vs ch1 ASR)" como pendiente (item P2 del roadmap), y la asignación física FL→Conference/FR→ASR del firmware concreto flasheado **no está confirmada empíricamente**. La firma de la memoria avisa: varias afirmaciones de estas sesiones fueron desmentidas por datos.
- **Cambio concreto:** probar `indata[:,1]` **per-device, no global** (el callback sirve también al mic UAC1.0 mono del escritorio — un swap ciego revienta con IndexError ahí). Medir **RMS y WER A/B**. Reversible (un índice).
- **Impacto:** alto si el canal ASR es el correcto. **Esfuerzo:** bajo. **No hacerlo a ciegas — es un experimento, no un hecho.**

#### L-4 — Fijar el beam hacia la zona de escucha (si el DoA es inestable)
- **Estado:** ⚠️ **A validar con datos** (re-medir DoA en silencio total primero).
- **Mecánica:** `AEC_FIXEDBEAMSONOFF 1` + `AEC_FIXEDBEAMSAZIMUTH_VALUES <rad>` fija 1-2 direcciones (sillón/escritorio) en vez de barrer 360° y captar TV/secador de otras direcciones. `AEC_FIXEDBEAMSGATING 1` silencia el beam sin voz (gate hardware extra, complementario a SPENERGY).
- **Por qué "a validar":** el DoA procesado es **inestable** (moda ~165.91° pero salta a 332°/282°/267° con ruido) — descartado como gate único. Solo vale fijar el beam **si** al re-medir en silencio el DoA se estabiliza hacia una zona consistente. Requiere L-1.
- **Impacto:** medio. **Esfuerzo:** medio.

### Lever de barge-in (esfuerzo medio-alto, condicional)

#### L-5 — Rutear el TTS por el playback USB del board para habilitar AEC por hardware
- **Estado:** mecanismo **confirmado**, beneficio **condicional** (requiere cambio de topología de audio).
- **Hallazgo:** el AEC del XVF3800 usa el canal izquierdo (0) de la salida USB como referencia far-end, **automáticamente**. Hoy el TTS/música de KZA salen por el **Dayton MA1260**, NO por el board → el AEC del chip está **ciego**, no puede cancelar ni el propio TTS ni Spotify ni la TV. Por eso KZA hace AEC en host por correlación numpy (`echo_suppressor.py`), que es frágil far-field.
- **Cambio:** rutear el playback de TTS también al sink USB del board (pipeline mono) habilitaría barge-in real con double-talk por hardware. Parámetros relevantes: `AUDIO_MGR_REF_GAIN`, `AEC_FAR_EXTGAIN`, `AUDIO_MGR_SYS_DELAY` (alinear referencia vs mic). Patrón probado por HA Voice PE (rutea todo por el chip + wake word "Stop" dedicado).
- **Límite duro honesto:** la **TV es fuente externa** — el AEC nunca tendrá su referencia. Para la TV no hay AEC posible; eso se ataca por SPENERGY + speaker-verification + colocación lejos de la TV, no por AEC.
- **Impacto:** medio (barge-in sobre TTS/música propios). **Esfuerzo:** medio-alto (toca routing de audio). Condicional a si vale la pena mover el audio del MA1260.

### Defensa contra audio-de-fondo-con-voz

#### L-6 — Speaker-verification (ECAPA-TDNN) como gate opt-in
- **Estado:** ⚠️ **Incierto / a validar con datos** (mecanismo existe; eficacia far-field sin medir).
- **Por qué:** ni SPENERGY ni Silero filtran la voz de videos/YouTube/TV — **es voz humana real**. El AEC tampoco (fuente externa). La defensa correcta es discriminar **quién** habla. KZA ya tiene el modelo (`src/users/speaker_identifier.py`, ECAPA-TDNN cuda:1) y el flag `security.require_known_speaker_for_actions` (`settings.yaml:490`, default off, cableado en `dispatcher.py:763-778`).
- **Advertencias serias (por eso "a validar"):**
  - El gate hoy **solo cubre el fast-path de domótica** (`PathType.FAST_DOMOTICS`), no música/Spotify, ni el slow/LLM path, ni rutinas. Voz de TV ruteada a música o al LLM pasaría igual.
  - **No hay voces enroladas** (`data/users.json` no existe). Con el flag en true sin enrollment, se rechaza **TODO** (incluido el dueño → "No te reconozco" en tu propia casa). El enrollment es bloqueante, no "costo menor".
  - Degradación far-field documentada: ECAPA pierde precisión con audio garbleado y utterances cortos (comandos de domótica son cortos) → sube tanto false-accept (TV pasa) como false-reject (dueño rechazado). Trade-off FAR/FRR real, no win gratis.
  - El "-38% de false positives" que se citó es de **voice-adaptation del detector** (entrenar a tu pronunciación), **no** de speaker-verification rechazando voces no-enroladas. No conflacionar.
- **Impacto:** alto si funciona; incierto en la práctica. **Esfuerzo:** medio (enrolar + extender el gate a más paths). **Validar:** medir false-reject del dueño far-field antes de flippear.

### NO hacer (callejones ya medidos — incluidos como advertencia)

#### N-1 — NO meter noise suppression en software (RNNoise/DeepFilterNet) global antes de Whisper
- **Estado:** ✅ **Confirmado** (papers + datos KZA).
- Evidencia académica convergente (ICAART 2024; arXiv 2512.17562 "When De-noising Hurts" — degradación en **todas** las 40 configs probadas, incluido large-v3): en Whisper **grande** la supresión SW **empeora el WER** (distorsión imperceptible al humano pero tóxica para el ASR). El DSP del XVF3800 ya hace dereverb+NS mejor y específico para ASR. Si alguna vez querés NS extra, solo en el peor caso (TV fuerte) y midiendo WER antes/después, nunca global a ciegas.

#### N-2 — NO activar enforce_confidence basado en confianza acústica de Whisper; mantener command_gate en shadow
- **Estado:** ✅ **Confirmado** (14.288 eventos instrumentados).
- `no_speech_prob=0.00` en **todos** los eventos (aceptados y rechazados) → inservible. `avg_logprob` **INVERTIDO**: alucinaciones median −0.42 (mejor) vs comandos reales −0.81 (peor) → imposible separar por umbral. Causa: turbo = 4 decoder layers; Calm-Whisper (arXiv 2505.12969) localiza el bug en "crazy heads" del decoder. `enforce_confidence:false` ya es el estado deployado correcto.
- **Guard alternativo barato que SÍ funciona:** filtrar post-hoc segmentos con `compression_ratio > 2.0-2.2` (práctica validada en openai/whisper #2378; el threshold built-in no funciona, hay que hacerlo post-transcripción).

#### N-3 — NO volver a Porcupine; NO usar DoA ni Silero como gate único
- Porcupine: free tier EOL 30/06/2026, `.ppn` caduca a 30 días, AccessKey valida online → **rompe el 100% on-prem**. Queda cableado pero inerte.
- DoA como gate único: ángulo inestable (ver L-4).
- Silero sobre el stream del XVF3800: devuelve prob≈0 sobre voz fuerte porque el chip ya hace DSP. Sí sirve recortado a un clip corto post-wake (endpointing), no sobre todo el audio.

---

## 3. reSpeaker Flex vs lo instalado

**Veredicto: no migrar. El Flex no mueve la aguja.**

El reSpeaker Flex (lanzado 2026-04-09, real, no confusión) usa **exactamente el mismo silicio (XMOS XVF3800) y el mismo firmware** que el array USB que KZA ya tiene. El motor acústico — AEC, beamforming, DoA, AGC, VAD, noise-suppression, dereverb — es **idéntico**. Las únicas diferencias son físicas y no tocan el cuello de botella:

| Aspecto | XVF3800 USB (instalado) | reSpeaker Flex | ¿Mueve la aguja? |
|---|---|---|---|
| Chip / firmware | XMOS XVF3800 / v3.2.x | **El mismo** | No |
| Algoritmos DSP | AEC/beam/DoA/AGC/VAD/NS/dereverb | **Idénticos** | No |
| Form factor | Placa integrada, USB plug&play | Split FPC (mics separables del core) | Solo si necesitás empotrar mics lejos |
| Array | Circular-4 (360°) | Circular-4 **o Linear-4 (~180°, supresión trasera)** | **Tal vez** (ver abajo) |
| Salida de audio | Sin amplificador de potencia | TPA3139D2 10W + AUX | No (TTS sale por MA1260) |
| Socket XIAO/WiFi/BT | No | Sí | No (KZA usa USB cableado) |
| Precio / disponibilidad | ~USD 49.90, ya instalado | ~USD 69.99 (Linear+XIAO), **pre-order 6-8 semanas**, alimentación 12V externa | Costo + tiempo por el mismo motor |

**La única ventaja física defendible** sería la variante **Linear-4** (~180° con supresión trasera por geometría) **si** —y solo si— tras agotar el tuning del chip, la fuente de ruido (TV) está **consistentemente DETRÁS** del mic. Eso atacaría el SNR de entrada *antes* de cualquier algoritmo, algo que el software no recupera. Pero es un **experimento de SNR, no una bala de plata**, y solo después de L-2/L-3/L-4.

**⚠️ Corrección a la investigación:** la frase atribuida a "los devs de FutureProofHomes" (que el XVF3800 "works MUCH BETTER") la escribió el **autor de la pregunta** en una discusión GitHub sin respuestas (un contribuidor de la comunidad HA, no un dev de FPH), y es especulativa ("will work"). La conclusión sigue siendo correcta, pero no la cites como endorsement de FPH.

---

## 4. Alternativas del mercado

**Veredicto: nada supera al XVF3800 para este caso.** Es la generación de DSP de voz **más nueva** de XMOS (User Guide v3.2.1, oct-2024); **no existe XVF4000 ni sucesor anunciado**. El consenso de la comunidad HA/maker 2025-2026 lo llama "one of the best mic arrays" para voz local con Whisper.

| Hardware | Chip / mics | DSP de voz | Veredicto vs XVF3800 |
|---|---|---|---|
| **reSpeaker XVF3800 USB (instalado)** | XVF3800 / 4 | AEC+beam+DoA+AGC+VAD+NS+dereverb, 360°/~5m | — (referencia) |
| reSpeaker Flex | XVF3800 / 4 | Idéntico | = chip, solo form factor (§3) |
| HA Voice PE | XU316 / **2** | AEC+NS estacionario+AGC, sin beamformers múltiples | **Paso atrás** (los propios devs FPH barajan el XVF3800 como upgrade). Único plus: firmware open-source (ESPHome) |
| FutureProofHomes Satellite1 | XU316 / 2 activos | Stack liviano | Paso atrás |
| reSpeaker Lite | XU316 / 2 | Sin beamforming/DoA fuerte | Paso atrás ("salas grandes ruidosas" es justo lo que cubre el XVF3800) |
| miniDSP UMA-8 v2 | **XVF3000** (viejo) / 7 | Beam+AEC+NS genéricos | Plataforma en fin de vida; reviews recomiendan migrar al XVF3800 |
| miniDSP UMA-16 | sin DSP de voz / 16 | **Ninguno** (16 canales crudos, procesás vos) | Solo para beamforming propio en GPU (no es el caso; DoA descartado) |
| Anker PowerConf S330 / Jabra Speak | 4 / varios | AEC caja-negra, no ajustable | Radio ~3m, tuning telefónico que rompe ASR. Solo útil como mic de escritorio cercano (mejoraría el UAC1.0 genérico del escritorio) |
| PS3 Eye / Matrix Voice / ESP32-S3-BOX / Korvo | varios / 2-4 | Crudo o muy inferior; Matrix Voice **descontinuado** | Todos por debajo del HA Voice PE → muy por debajo del XVF3800 |

**Conclusión:** tenés razón, "son muy buenos". El silicio ya da lo que el caso necesita; el success ~12% **no se explica por el mic** (se explica por AGC sin calibrar + referencia AEC ausente + gates/umbrales/acciones-fantasma). El de mayor ROI es exprimir el XVF3800, no comprar otro array.

---

## 5. Plan de acción propuesto

Ordenado por ROI y dependencias. Cada paso dice qué se valida.

**Fase 0 — Garantías operativas (esta semana, bajo riesgo, sin tocar el chip)**
1. **QW-2:** pinear `pyusb==1.3.1` en `requirements.txt`; verificar que la udev rule `MODE=0666` esté aplicada en el server. *Valida:* que el gate SPENERGY está realmente ON en prod (chequear logs `[SPENERGY-gate]`, no asumir).
2. **QW-1:** cerrar el bypass del gate en early_dispatch (opción b). *Valida:* con TDD local — test que asegure que el gate corre también en el path early_dispatch.
3. **QW-3:** cablear `silence_threshold`/`silence_duration_ms`/`min_speech_ms` al loop. *Valida:* endpointing con umbrales del XVF3800, no defaults genéricos. Cuidado con la semántica `min_rms` ≠ `silence_threshold`.
4. **N-2 (guard):** agregar filtro post-hoc `compression_ratio > 2.0-2.2`. *Valida:* contra las frases-basura ("Gracias", "Suscríbete al canal") sin tocar enforce_confidence.

**Fase 1 — Front-end configurable (esfuerzo medio, en RAM, reversible)**
5. **L-1:** extender `XvfController` con write USB (simétrico al read, fail-open). *Valida:* leer-modificar-leer un parámetro inocuo primero (p.ej. `PP_AGCMAXGAIN`) para confirmar el protocolo en el chip real antes de tocar nada crítico.
6. **L-2:** bajar el AGC (elegir vía A o B, **no ambas**). *Valida:* medir RMS del piso de ruido y success rate A/B. En RAM, **nunca `save_configuration`**. Si la voz far-field queda baja, subir el techo a 12-16.
7. **L-3:** A/B canal 0 vs canal 1, **per-device**. *Valida:* RMS + WER comparativo. Es experimento, no hecho — confirmar cuál canal es cuál en el firmware flasheado.

**Fase 2 — Filtrado direccional y barge-in (condicional, tras Fase 1)**
8. Re-medir DoA en **silencio total**. *Valida:* si se estabiliza, **L-4** (fijar beam hacia la zona de escucha). Si sigue saltando, no fijar.
9. **L-5** (opcional): evaluar rutear TTS por el playback USB del board para AEC/barge-in por hardware. *Valida:* solo si vale mover audio del MA1260; recordar que la TV externa nunca tendrá referencia AEC.
10. **L-6** (opcional): enrolar voces + extender el gate de speaker-verification a más paths antes de flippear. *Valida:* false-reject del dueño far-field (que no te rechace en tu casa).

**Qué NO hacer (ya descartado con datos — no re-gastar esfuerzo):**
- **No** meter NS en software (RNNoise/DeepFilterNet) global antes de Whisper (N-1).
- **No** activar enforce_confidence por confianza acústica de Whisper (N-2).
- **No** volver a Porcupine, ni usar DoA o Silero como gate único (N-3).
- **No** migrar a reSpeaker Flex circular (mismo chip, USD inútil). El Linear-4 solo como experimento de SNR si la TV está detrás del mic, y solo después de la Fase 1.
- **No** comprar otro array: el XVF3800 es el tope del mercado para este caso.

**Nota de método (de la propia memoria del proyecto):** varias afirmaciones de estas sesiones fueron desmentidas por los datos. Todo lo marcado "a validar con datos" arriba se mide A/B antes de afirmarlo. El server lidera; verificar con `git merge-base` y `git hash-object` antes de tocar, en producción read-only.

---

### Fuentes principales
- **Código KZA:** `src/audio/xvf_controller.py:33-171`; `src/pipeline/multi_room_audio_loop.py:32-47,95-97,329-356,383,522-544`; `src/main.py:828-861`; `src/orchestrator/dispatcher.py:763-778`; `config/settings.yaml:1011-1212`; `deploy/udev/99-xvf3800.rules:10`; `requirements.txt`.
- **Docs KZA:** `docs/research/2026-05-31_SESION_CONSOLIDADA_wake_alucinaciones.md`; `docs/research/2026-05-31_DIAGNOSTICO_ALUCINACIONES_Y_ROADMAP_ALEXA.md`; `docs/research/2026-05-31_BUG_RAIZ_WAKE_NEXA_Y_ROADMAP_v2.md`.
- **XMOS oficial:** `xmos.com/.../04_tuning_the_application.html`, `.../03_audio_pipeline.html`, `.../AA_control_command_appendix.html`, `.../CC_alternative_tuning_parameters.html`, User Guide v3.2.1.
- **Seeed / repo:** `wiki.seeedstudio.com/respeaker_xvf3800_introduction/`, `wiki.seeedstudio.com/respeaker_flex_introduction/`, `github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY` (`python_control/xvf_host.py`, issue #8).
- **Académicas:** ICAART 2024 (scitepress 124571); arXiv 2512.17562, 2505.12969 (Calm-Whisper); openai/whisper #2378.
- **Mercado:** `seeedstudio.com` (XVF3800/Flex blogs), `github.com/FutureProofHomes/Satellite1-Hardware/discussions/40`, `minidsp.com`, `xda-developers.com`, `smarthomecircle.com`.