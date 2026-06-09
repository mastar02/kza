# Sesión consolidada — Wake, alucinaciones y XVF3800 (2026-05-31)

> Documento de cierre de sesión. Separa **lo PROBADO con datos** de **lo DESCARTADO**
> y deja el plan para retomar. Toda la investigación fue **read-only** sobre el server
> `kza@192.168.1.2` (producción) — **no se modificó nada**: AGC en sus valores
> originales (64/on), `kza-voice` activo, chip intacto.
>
> ⚠️ **Nota de método:** durante la sesión afirmé varias conclusiones con seguridad que
> los datos luego desmintieron (ver §"Correcciones"). Todas se corrigieron al verificar.
> Este doc contiene **solo lo verificado**, con su evidencia.

---

## TL;DR

1. **El wake "Nexa" NO está roto.** En producción dispara bien (scores 0.7-0.9 al decirlo;
   0.000 en silencio). El "1 de 7" percibido viene de **(a) el comando far-field se
   transcribe garbleado** y **(b) audio de fondo con voz** (videos) que confunde a Whisper.
2. **HALLAZGO DE ORO:** el XVF3800 tiene un **detector de voz por hardware** (`SPENERGY`,
   pre-AGC, 0-GPU) que **distingue tu voz del ruido del secador con separación enorme**
   (voz 101–2.1M vs secador 0). Es la palanca más fuerte para matar las alucinaciones.
3. **El AGC sí está activo y amplificando** (`PP_AGCGAIN=8.06`, `MAXGAIN=64`): contribuye
   a elevar ruido/silencio. Bajarlo es una mejora válida y reversible.
4. **Porcupine descartado** (free tier de Picovoice cierra 30/06/2026 + valida online).
5. El **filtro direccional (DoA)** resultó **menos confiable** de lo esperado (el ángulo
   salta mucho con ruido/reverb). Queda en duda hasta re-medir en silencio.

---

## 1. Estado real del wake (PROBADO)

Verificado en el server (read-only):
- `engine: openwakeword`, `model: /home/kza/kza/models/wakeword/nexa.onnx`, `threshold: 0.3`.
  `kza-voice` activo (PID 3716370). HEAD server `224912d`, local `75d4685` (drift).
- **Dispara en vivo:** logs `Wake word in escritorio (nexa: 0.84 / 0.76 / 0.67 / 0.55…)`;
  silencio = `0.000` (cero FP). Comandos OK llegan a HA
  (`[HA-CALL] light.turn_on@light.escritorio_2 success=True`).
- **A/B (CPU):** recall **75% holdout / 93% train**, silencio 0.002.
- El detector del server YA carga el `.onnx` por la vía correcta de openWakeWord. El "bug
  del grafo crudo" (`_predict_custom`) **solo existe en la copia LOCAL vieja**, no en prod.

**Por qué el "1/7" igual es real:** el wake dispara, pero el **STT del comando far-field
garblea** (ej.: `'Nexa, apagarendo directo'`, `'Nexa, abrené le lluvia le lluvia'`). Solo
cuando el comando sale limpio (`'Nexa, prendé la luz del escritorio'`) la acción ocurre.

---

## 2. HALLAZGO DE ORO — SPENERGY = VAD por hardware (PROBADO)

Medido con el secador prendido (que es tu condición habitual):

| Fuente | SPENERGY (beam auto-select) |
|---|---|
| Secador solo (sin tu voz) | **0** (120/120 lecturas) |
| Tu voz "Nexa" | **101 – 2.121.738** (mediana ~29.000) |

El detector de voz del DSP del XVF3800 **ignora el ruido de banda ancha** (secador) y
**marca la voz humana**, operando **antes del AGC**. Separación limpísima → un pre-gate
con umbral ~50-100 dejaría pasar tu voz y bloquearía el secador/silencio.

**Implicancia:** correr este pre-gate **antes de Whisper** eliminaría las alucinaciones de
"Gracias"/"Gracias por ver el video" sobre el secador y el silencio, **sin gastar GPU**.

**Cómo se lee (verificado):** binario oficial `xvf_host` v3.0.0 (repo
`respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY`, en `/tmp/xvf3800_tool/`), Interface 3
(vendor, no toca el audio streaming). `pyusb 1.3.1` instalado en `~/app/.venv`.
Sampler rápido: `xvf_host -e <lista>` → ~62 lecturas/s. Parámetro: `AEC_SPENERGY_VALUES`
(4 floats, índice 3 = beam auto-select). Firmware USB v2.0.6.

---

## 3. AGC — activo y amplificando (PROBADO, corrige al diagnóstico)

Lectura directa del chip:
```
PP_AGCONOFF       = 1        (encendido)
PP_AGCGAIN        = 8.06     (amplificando ×8 ahora)
PP_AGCMAXGAIN     = 64       (hasta ×64 en silencio)
PP_AGCDESIREDLEVEL= 0.0045
AUDIO_MGR_MIC_GAIN= 90
AEC_FAR_EXTGAIN   = -43 dB   (referencia far-end casi muteada → AEC ciego, sin barge-in)
```
El AGC eleva el nivel del audio (incl. ruido en los huecos de voz). **Bajar
`PP_AGCMAXGAIN 64→8`** (en RAM, reversible desenchufando, sin `save_configuration`) es una
mejora complementaria de bajo riesgo. NO es la causa única de las alucinaciones.

---

## 4. Lo DESCARTADO o en duda

- **Filtro direccional (DoA):** el ángulo procesado (`AUDIO_MGR_SELECTED_AZIMUTHS[0]`)
  **no es estable** con ruido/audio de fondo: moda 165.91° pero salta a 332°, 282°, 267°…
  Re-medir en silencio total antes de apostar a esto. Probablemente no alcance solo.
- **Porcupine:** descartado (free tier EOL 30/06/2026, `.ppn` caduca 30 días, AccessKey
  valida online → rompe 100% on-prem).
- **VAD software (Silero) como única solución para el audio de fondo:** NO sirve — el
  audio de fondo (videos) ES voz humana, el VAD no la filtra. Para eso: SPENERGY +
  speaker-verification o beam.
- **Señales de confianza de Whisper** (`no_speech_prob`, `avg_logprob`): muertas (turbo
  4 layers). Confirmado por arXiv 2501.11378 y 2505.12969.

---

## 5. Plan para retomar (ordenado por impacto/evidencia)

### Alto impacto, verificado
1. **Pre-gate SPENERGY antes de Whisper.** Crear `src/audio/xvf_controller.py` (pyusb, ya
   instalado) que lea `AEC_SPENERGY_VALUES`; en `multi_room_audio_loop.py`, no transcribir
   si SPENERGY[3] < umbral (~50-100). Mata alucinaciones de secador/silencio, 0-GPU.
   Implementar en LOCAL con TDD; deploy con OK del usuario (comparar hash, preflight VRAM).
2. **Bajar AGC `PP_AGCMAXGAIN 64→8`** (RAM, reversible). 1 comando. Mejora el SNR de la voz.

### Para el audio-de-fondo-con-voz (el caso difícil del usuario, "casi siempre hay audio")
3. **Speaker-verification opt-in** (`security.require_known_speaker_for_actions`, ya existe):
   solo obedece la voz enrolada del usuario → el video de fondo no dispara acciones.
   Requiere enrolar la voz.
4. (Investigar) **beam fijo** hacia la silla (`AEC_FIXEDBEAMS*`) si el DoA se estabiliza en
   silencio — re-medir primero.

### UX / latencia (del informe del workflow, válido, no urgente)
5. Earcon <200ms al wake (feedback inmediato). Ring prebuffer + endpointing 500→200ms.
6. Expandir gramática determinística (evita el LLM router de 799ms). Mover Whisper STT a CPU
   (libera ~1.5GB cuda:1, mata el OOM).

### Mantenimiento
7. Quitar instrumentación `[oww-dbg]` de `detector.py:233` (server).
8. (Opcional) sync del fix de `detector.py` local→server para no re-introducir el bug viejo.

---

## 6. Correcciones hechas durante la sesión (transparencia)

| Afirmé (mal) | Realidad verificada |
|---|---|
| "El wake da 1/7 por un bug del grafo crudo en detector.py" | El server YA carga bien; el wake funciona. El bug solo está en la copia local vieja. |
| "El AGC ya está apagado" | `PP_AGCONOFF=1`, amplificando ×8. Activo. |
| "Tu voz = DoA 208°, energía 0.021" | Inventado; el SPENERGY estaba en 0 (sampler lento). Falso. |
| "DoA fijo en 165.91°" | Es la moda, pero salta mucho (332°, 282°…). No estable. |

**Causa del patrón:** producir conclusiones antes de mirar el dato. Corrección operativa:
verificar SIEMPRE antes de afirmar; en producción, read-only + confirmar antes de tocar.

---

## Archivos de esta sesión
- Memoria: `project_wake_word_dedicado_f2_2026-05-31.md` (bloques sesión 2 y 3).
- `docs/research/2026-05-31_BUG_RAIZ_WAKE_NEXA_Y_ROADMAP_v2.md` (roadmap corregido).
- Este doc (cierre).
- Tooling XVF3800 en server: `/tmp/xvf3800_tool/` (repo oficial + `xvf_host` + firmware 6ch).
- `pyusb 1.3.1` instalado en `~/app/.venv` (único cambio persistente, inerte).
