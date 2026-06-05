# Compuerta acústica integral — AmbientGuard + calibración medición-primero

**Fecha:** 2026-06-05
**Branch:** `feat/nexa-command-detection-fixes`
**Estado:** Diseño aprobado

## Problema

Con TV de fondo, todas las compuertas acústicas del pipeline están rotas o apagadas
(estado verificado en código al 2026-06-05):

| Compuerta | Estado | Causa |
|---|---|---|
| Gate SPENERGY (`xvf_controller.py`) | `enabled: false` | Lee pico 0 con voz real desde que el chip corre MAXGAIN=8; la calibración del 31-05 (umbral 100) se hizo con MAXGAIN=64 y es inválida. |
| Silero VAD (interno faster-whisper y standalone) | OFF | prob~0 sobre la salida DSP del XVF3800; el interno borraba capturas enteras con voz (fix `1812eaa`). |
| `min_wake_rms` | `0.0` (OFF) | Nunca calibrado post-AGC-8. |
| Wake openwakeword `nexa.onnx` @ 0.40 | ON pero roto con TV | Dispara 0.4-0.9 constantemente con TV de fondo. |

Además, el TV-mode existente vive **solo** en `WhisperWakeDetector` (`whisper_wake.py`),
el engine inactivo. El path activo (openwakeword → `multi_room_audio_loop`) no tiene
ningún modo adaptativo.

Resultado (noche 06-04/06-05): el wake es la única puerta acústica y con TV está abierta
→ wakes espurios en cascada, follow_up siempre activo, LLM router saturado (timeouts 3.5s
en cadena), 1 acción fantasma (contenida con guardias de longitud, commit `04772de`), y
"no me escucha" = comandos reales compitiendo contra la cola.

## Decisiones de alcance (usuario, 2026-06-05)

- **Sesión en vivo completa disponible** (server + mic + TV + voz del usuario).
- **Prioridad: falsos positivos.** Cero acciones fantasma y cero saturación; se acepta
  repetir un comando o hablar más fuerte con TV prendida.
- **Re-entrenamiento del wake FUERA de alcance** — proyecto aparte. Esta sesión: solo
  compuertas con el modelo actual. (Los datos crudos de calibración quedan como insumo
  para ese futuro entrenamiento.)

## Principio rector

**Medición antes que umbral.** Las sesiones del 31-05 al 04-06 demostraron repetidamente
que los umbrales adivinados mueren contra los datos (SPENERGY "pre-AGC" desmentido,
calibración invalidada por cambio de MAXGAIN, avg_logprob invertido). Ningún umbral
acústico se cablea sin distribución medida con el chip en su estado actual
(MAXGAIN=8, ch1 ASR, `vad_filter: false`).

## Arquitectura

### AmbientGuard — escalera de 3 estados, por habitación

Componente nuevo `src/pipeline/ambient_guard.py`. Unifica TV-mode + circuit breaker
en una máquina de estados por room. Supersede al TV-mode del whisper-wake (que no se toca:
código del engine inactivo).

```
                 wakes rechazados/min ↑          rechazos persisten ↑
   ┌─────────┐  (o piso acústico ↑)   ┌─────────┐  (saturación inminente) ┌──────────┐
   │ NORMAL  │ ─────────────────────► │ STRICT  │ ───────────────────────►│ COOLDOWN │
   │         │ ◄───────────────────── │(TV-mode)│ ◄─────────────────────── │ (breaker)│
   └─────────┘   quiet sostenido      └─────────┘   ventana cumplida       └──────────┘
                 (histéresis)
```

| Estado | Wake | Compuertas | Follow-up |
|---|---|---|---|
| **NORMAL** | threshold base (0.40) | calibradas (laxas) | activo |
| **STRICT** | threshold elevado (de la matriz, ~0.6-0.7) | `min_wake_rms` estricto; bonus `wake_acoustically_confirmed` deshabilitado (se exige "nexa" en el texto transcripto) | desactivado |
| **COOLDOWN** | capturas descartadas con log, duración fija | — | — |

**Señales de entrada:**
- *Software (siempre disponible, no depende del chip):* tasa de capturas rechazadas
  (noise / texto vacío / timeout del router) por ventana móvil. Funciona igual en el
  living (mic UAC1.0 sin XVF3800).
- *Acústicas (solo si la matriz las valida):* piso de RMS sostenido, piso de SPENERGY.

**Decisiones de diseño:**
1. El detector openwakeword queda fijo en 0.40; AmbientGuard aplica el threshold elevado
   **encima**, en `_should_accept_wakeword()` (`multi_room_audio_loop.py:224`). No se
   muta el detector; la decisión vive en un solo lugar testeable.
2. COOLDOWN garantiza "nunca más saturado" por construcción: M rechazos consecutivos en
   T segundos cortan el flujo antes de que la cola del router se llene, sea cual sea el
   destino de las señales acústicas.
3. Estado por room (el escritorio tiene XVF3800; el living no — señales disponibles
   difieren por construcción).
4. Config en `settings.yaml` bajo `rooms.ambient_guard.*`; defaults = comportamiento
   actual exacto (guard pasivo hasta flip explícito — patrón shadow→enforce de CommandGate).

### Interfaz

```python
@dataclass
class GuardDecision:
    accept: bool
    reason: str            # "ok" | "strict_threshold" | "cooldown" | ...
    state: GuardState      # NORMAL | STRICT | COOLDOWN

class AmbientGuard:
    def on_wake(self, room, score, rms, spenergy_peak) -> GuardDecision
    def on_capture_result(self, room, outcome)  # accepted|noise|empty|timeout
    def state_for(self, room) -> GuardState
    def follow_up_allowed(self, room) -> bool
```

Clase pura: sin I/O, decisiones síncronas, reloj inyectado (`time_fn`) para testear
ventanas e histéresis sin sleeps.

**Integración en `multi_room_audio_loop`:**
- `_should_accept_wakeword()` consulta `guard.on_wake(...)` (ya recibe score y rms).
- Resultado de cada captura → `guard.on_capture_result()`, hook nuevo al final de
  `_dispatch_command` (los 3 outcomes ya se loggean hoy).
- `FollowUpMode.start_conversation()` condicionado a `follow_up_allowed()`.

## Harness de calibración — `tools/acoustic_calibration.py`

Standalone, se corre **con kza-voice parado** (contención de mic + USB). Por condición
etiquetada captura ~2-3 min simultáneamente:

| Señal | Fuente | Frecuencia |
|---|---|---|
| RMS por chunk | stream del mic, mismo device/canal que prod (ch1 ASR, binding `mic_usb_port`) | por chunk (~80ms) |
| Score wake | `nexa.onnx` openwakeword sobre el mismo stream | por frame |
| SPENERGY[3] | `XvfController.read_param("AEC_SPENERGY_VALUES")` | poll 25Hz |

**Protocolo (usuario en el escritorio):**
1. `silencio` — 2 min, TV apagada, nadie habla
2. `tv` — 3 min, TV a volumen normal, nadie habla
3. `voz` — TV apagada, ~10 comandos reales con "Nexa" a distancia normal
4. `voz_tv` — TV prendida, ~10 comandos reales

**Salida:** JSONL crudo (`data/calibration/2026-06-05_*.jsonl` — queda como dataset para
el futuro re-entrenamiento del wake) + tabla p5/p50/p95/max por señal por condición.

**Pregunta que responde:** ¿qué señal tiene gap entre `voz`(p5) y `tv`(p95)? Esa señal es
compuerta viable; las que no separan quedan documentadas como muertas (también es un
resultado: deja de haber candidatos zombie).

## Compuertas recalibradas (decisión post-matriz)

- **SPENERGY:** si separa → `spenergy_gate.enabled: true` con umbral medido. Si lee 0 con
  voz (como el 06-04) → veredicto final "muerto con AGC=8", se retira de candidatos y el
  yaml lo documenta.
- **`min_wake_rms`:** ídem. Advertencia conocida: el AGC comprime niveles — puede que RMS
  no separe voz de TV; por eso se mide antes de cablear.
- **Threshold STRICT del wake:** de la distribución de scores `tv`(p95) vs `voz`(p50).
  Si la TV alcanza 0.9 con frecuencia, el threshold solo no alcanza — la escalera
  (señal de software) carga el peso, y no depende de eso.

## Manejo de errores

- **XvfController ausente/falla** → fail-open como hoy: señales acústicas omitidas, la
  escalera funciona solo con la señal de software. Un fallo USB nunca deja el pipeline sordo.
- **Living (sin chip)** → mismo AmbientGuard, señales acústicas ausentes por construcción.
- **Config ausente** → defaults = comportamiento actual exacto (guard pasivo).
- **COOLDOWN nunca silencioso:** cada captura descartada se loggea con estado y contadores
  (`[AmbientGuard]`), auditable como `[CommandGate]`.

## Testing

- **Unit (lo grueso):** máquina de estados — entrada/salida de STRICT con histéresis,
  escalada a COOLDOWN, expiración de ventanas con reloj inyectado, aislamiento per-room,
  follow_up bloqueado en STRICT, defaults pasivos.
- **Integración:** `_should_accept_wakeword` respeta `GuardDecision`; `on_capture_result`
  cableado en los 3 outcomes.
- **Análisis de calibración:** percentiles y recomendación de umbral con distribuciones
  sintéticas conocidas. El loop de captura es glue fino sin test.
- Nota laptop: módulos con mocks de `sys.modules` (torch) se corren aislados
  (interferencia conocida en suite completa).

## Plan de sesión (2026-06-05)

1. **Local (TDD):** funciones de análisis + harness → deploy al server.
2. **En vivo (~30-45 min, kza-voice parado):** matriz de 4 condiciones. Usuario habla;
   harness controlado por SSH.
3. **Análisis:** tabla de gaps → decisión conjunta de qué señales se cablean y umbrales.
4. **Local (TDD):** AmbientGuard completo + integración + umbrales en `settings.yaml`.
5. **Deploy + validación en vivo.** Commit por paso en `feat/nexa-command-detection-fixes`,
   `kza-push`.

Si la sesión en vivo se corta tras el paso 3, el diseño no se invalida: AmbientGuard con
solo la señal de software ya resuelve la saturación; las compuertas acústicas se suman después.

## Protocolo de validación en vivo (criterio de éxito)

| Escenario | Esperado |
|---|---|
| TV prendida 10 min, nadie habla | 0 acciones, 0 timeouts del router en cadena; guard llega a STRICT y no oscila |
| Comando con TV (voz firme) | funciona, o rechazo honesto y al repetir más cerca funciona |
| TV apagada, quiet 2-3 min | guard vuelve a NORMAL; comandos 3/3 (paridad con ronda 4 del 06-04) |
| Apagar TV en medio de STRICT | salida por histéresis sin quedarse pegado |

## Fuera de alcance

- Re-entrenamiento del wake `nexa.onnx` (proyecto aparte; esta sesión le deja dataset).
- A/B de `AEC_ASROUTGAIN` 1→2-4 (palanca de garbles far-field, no de compuerta) — solo
  si sobra sesión, fuera del core.
- Keepalive del WS HA, XvfController por-room (living sin mic), calibración de
  `compression_ratio` — pendientes menores preexistentes, no este diseño.
