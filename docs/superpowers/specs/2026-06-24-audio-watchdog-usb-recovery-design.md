# Auto-recuperación del stream de audio ante re-enumeración USB

- **Fecha:** 2026-06-24
- **Estado:** Diseño aprobado (brainstorming) — pendiente plan de implementación
- **Rama:** `feat/audio-watchdog-usb-recovery` (base `main @ ab479ac`)
- **Archivos afectados:** `src/pipeline/multi_room_audio_loop.py`, `src/rooms/room_context.py` (reuso), `config/settings.yaml`

## Problema

El 2026-06-21 19:39 el ReSpeaker XVF3800 se re-enumeró en el bus USB (el usuario
confirmó: el bus perdió corriente). El kernel borró los device nodes viejos y creó
nuevos (`pcmC1D0c`, `controlC1` recreados con timestamp 19:39; el devnum pasó de 5 a 8).
El proceso `kza-voice` —vivo ininterrumpidamente desde el 2026-06-14— **no detectó la
reconexión ni reabrió el stream de captura**: quedó con file descriptors colgados a
inodes borrados (`fd 100 -> /dev/snd/pcmC1D0c (deleted)`, `fd 80 -> /dev/bus/usb/003/005 (deleted)`).

Resultado: el proceso siguió "vivo" y el servicio `active`, pero **sordo durante 3 días**.
El `audio_callback` dejó de recibir frames (cero líneas `[oww-dbg]` en el journal en toda
la ventana), por lo que ningún wake/STT se procesó. El usuario pidió la luz ~6 veces sin
efecto. Se resolvió a mano reiniciando el servicio (`systemctl --user restart kza-voice`),
que re-resolvió el device por puerto y reabrió el stream.

**No existe auto-recuperación de re-enumeración USB en el pipeline.** Va a volver a pasar.

## Objetivo

Que el pipeline **detecte automáticamente** cuando el micrófono deja de entregar audio
(por re-enumeración USB u otra causa que mate el stream) y **reabra el stream sobre el
device actual**, sin intervención manual y sin reiniciar el servicio (modelos en GPU
siguen cargados, asistente no se corta).

### No-objetivos (fuera de scope de esta entrega)

- **Prevención** de la re-enumeración (udev rule para deshabilitar USB autosuspend del
  XVF3800). El incidente fue pérdida de corriente física, no autosuspend; requiere root
  en el server. Follow-up opcional.
- **Métrica** del evento al dashboard de obs `:9500`. Follow-up opcional; los logs ya dan
  visibilidad.
- **Deploy gap:** producción (server, `feat/nexa-command-detection-fixes @ 1c9669b`) corre
  10 commits atrás de `main`. Ponerla al día es un tema de deploy separado, no de esta feature.

## Causa raíz (medida, no asumida)

1. PortAudio toma el snapshot de devices en `Pa_Initialize` (al arrancar el proceso) y
   **nunca re-escanea**. Verificado: no hay ningún `sd._terminate()/_initialize()` en `src/`.
2. Tras la re-enumeración, el índice de PortAudio que el stream tenía abierto apunta a un
   device muerto. El callback deja de ser invocado.
3. El binding del mic ya está diseñado para sobrevivir esto **en el arranque**
   (`mic_usb_port` → índice vía `resolve_mic_usb_port`), pero esa resolución solo corre una
   vez al iniciar. Falta dispararla **en caliente** cuando el device cambia.

## Diseño

### Componentes

| Pieza | Ubicación (base local) | Responsabilidad |
|------|------------------------|-----------------|
| `RoomStream.last_frame_ts: float` | `multi_room_audio_loop.py:70` (dataclass) | Timestamp monotónico del último frame recibido. |
| Estampado en callback | `audio_callback`, `:570` | Primera línea: `rs.last_frame_ts = time.monotonic()`. O(1), fail-safe. |
| `detect_stale_streams(...)` | función pura nueva (módulo testeable) | Dado el estado y `now`, devuelve qué rooms superaron el timeout. Sin I/O. |
| `_stream_watchdog()` | task async en `MultiRoomAudioLoop` | Cada `check_interval_s` evalúa staleness; si hay stale → `_recover_streams`. |
| `_open_stream(rs)` | extraído de `run()` (`:452-466`) | Abre y arranca un `sd.InputStream` para un room. Reutilizable. |
| `_recover_streams(reason)` | método nuevo en `MultiRoomAudioLoop` | Reapertura coordinada (ver Recuperación). |

`resolve_mic_usb_port` (`room_context.py:862`) y `usb_port_to_alsa_card` (`room_context.py:802`)
se **reusan tal cual** (ya aceptan `devices` inyectable para tests).

### Detección

El callback corre en el thread C de PortAudio y estampa `last_frame_ts` en cada frame
(período normal ~50–80 ms). El `_stream_watchdog` —task async **dedicado**, arrancado en `start()` (`:360`)
junto a los demás tasks (análogo al loop de poll de `run()` en `:472`, pero independiente de él)— compara `monotonic() - last_frame_ts` contra
`no_frames_timeout_s` (default **8 s**). 8 s equivale a >100 períodos perdidos: inequívoco.
El stream nunca pausa legítimamente (sigue capturando durante TTS por el barge-in check),
así que no hay falsos positivos. La decisión vive en `detect_stale_streams`, función pura,
para testear sin hardware ni timing real.

### Recuperación (`_recover_streams`)

```
1. log ERROR "[audio-watchdog] mic <room> sin frames hace Xs → recuperando (reason)"
2. Para cada stream afectado: stream.stop(); stream.close()  (best-effort, ignora errores)
3. sd._terminate(); sd._initialize()        ← fuerza a PortAudio a re-escanear el bus
4. Espera al device: poll usb_port_to_alsa_card(mic_usb_port) con backoff
   reopen_backoff_min_s → reopen_backoff_max_s; reintenta indefinido mientras no aparezca
5. Re-resuelve índice: resolve_mic_usb_port(port, devices=sd.query_devices())
6. _open_stream(rs) para cada room afectado + reaplica XVF-tuning (AGC=8) si corresponde
7. log INFO "[audio-watchdog] mic <room> recuperado (device=N)"
```

**Por qué el paso 3 es obligatorio:** sin `_terminate()/_initialize()`, `sd.query_devices()`
devuelve la lista cacheada vieja y el reopen volvería a abrir el índice muerto. Reiniciar el
backend es **global a PortAudio**, por eso `_recover_streams` opera sobre **todos** los
streams activos de forma coordinada (hoy hay 1 room; queda correcto para N).

### Refactor mínimo justificado

El bloque de apertura del stream está inline dentro de `run()` (`:452-466`). Se extrae a
`_open_stream(rs)` para que la apertura inicial y la reapertura compartan código (DRY). Es el
único refactor; no se toca nada más del archivo (962 líneas).

## Manejo de errores y casos límite

- **Device ausente** (USB aún sin corriente): el paso 4 espera con backoff de forma
  indefinida en vez de crashear. El servicio sigue vivo (TTS/HA/ambient operan); solo el mic
  espera. **Evita agotar `StartLimitBurst=3/300s`** del unit.
- **Excepción no prevista** en `_recover_streams`: se propaga y el proceso muere →
  systemd `Restart=on-failure` (RestartSec=10s) reinicia limpio. Es la red de seguridad
  última y nunca deja el sistema peor que el estado de hoy.
- **Falso positivo:** descartado por el umbral; aun así el reopen es idempotente (si el
  device está sano, reabre el mismo índice y continúa).
- **Multi-room:** la reapertura es coordinada (paso 3 es global). Correcto para N streams.
- **Carrera con `stop()`:** el watchdog respeta `self._running`; al apagar el servicio no
  intenta recuperar.

## Configuración (nueva, en `config/settings.yaml`)

```yaml
audio:
  stream_watchdog:
    enabled: true
    no_frames_timeout_s: 8.0     # sin frames del callback por más de esto → recuperar
    check_interval_s: 2.0        # cada cuánto evalúa el watchdog
    reopen_backoff_min_s: 1.0    # espera inicial entre intentos de reabrir
    reopen_backoff_max_s: 10.0   # tope del backoff
```

`enabled: false` desactiva el watchdog (rollback de comportamiento sin revertir código).

## Testing

- **Unit (laptop/CI, sin hardware):**
  - `detect_stale_streams`: timestamps mockeados → detecta/ignora según umbral.
  - Re-resolución: `resolve_mic_usb_port` con `devices` inyectable simulando índice
    viejo→nuevo tras re-enumeración.
  - `_recover_streams`: con `sd` y `_open_stream` mockeados, verifica la secuencia
    (stop → terminate/initialize → espera → re-resuelve → reabre) y el backoff cuando el
    device está ausente.
- **Integración en hardware (server, coordinado con el usuario):** forzar re-enumeración del
  XVF3800 **sin tocar el cable** vía
  `echo 0 > /sys/bus/usb/devices/3-1.4/authorized; sleep 2; echo 1 > /sys/bus/usb/devices/3-1.4/authorized`
  y verificar en el journal que el watchdog detecta y recupera. Reproduce el evento del 21/6
  de forma controlada. ⚠️ Es producción → se ejecuta junto al usuario, nunca por iniciativa propia.

## Interfaces de las unidades nuevas

- **`detect_stale_streams(states: list, now: float, timeout_s: float) -> list[room_id]`**
  Qué hace: decide qué rooms están stale. Uso: la llama el watchdog cada tick.
  Depende de: nada (función pura) → totalmente testeable.
- **`_open_stream(rs: RoomStream) -> sd.InputStream`**
  Qué hace: abre y arranca un InputStream para un room. Uso: apertura inicial y reapertura.
  Depende de: `sounddevice`, `rs.device_index`, `sample_rate`, `CHUNK_SIZE`.
- **`_recover_streams(reason: str) -> None`**
  Qué hace: reabre coordinadamente todos los streams tras detección de mic muerto.
  Uso: la invoca el watchdog. Depende de: `_open_stream`, `room_context` (resolución por
  puerto), `sounddevice` (terminate/initialize).
- **`_stream_watchdog() -> None`** (task async)
  Qué hace: loop periódico que detecta staleness y dispara recuperación.
  Uso: arranca en `start()` junto al resto de tasks. Depende de: `detect_stale_streams`,
  `_recover_streams`, config.
