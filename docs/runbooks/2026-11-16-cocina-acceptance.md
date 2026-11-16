# Aceptación preregistrada de la Cocina (C19/C20)

Estado: **preregistrado, no ejecutado**. Lo escribió T025 (P2) el 2026-11-16 sin
tocar el servidor de producción. La ejecución física (micrófono XVF3800 de la
Cocina, su extensor Cat5e, el soak de 72 h y la inyección de fallos) queda para
P7, con un candidato congelado. Este guion se fija **antes** de mirar resultados:
los umbrales, las fallas a inyectar y el modo de cálculo no se cambian después.
Un cambio exige una versión nueva del guion, fechada, antes de la ventana.

Alcance: la entrada de audio de la Cocina y el efecto sobre las demás entradas.
La parte de C20 sobre HA/WebSocket (recuperación en <=60 s) no está en este
guion.

## Criterios

- **C19**: disponibilidad observada >=99,9 % de cada entrada requerida en un
  bloque físico fijo de >=72 h con el candidato congelado, dentro de siete días.
  Ninguna habitación queda compensada por las demás. No se excluyen fallos
  retrospectivamente. La inyección de fallos ocurre antes o después del bloque
  de estabilidad, nunca dentro. Se publican también la disponibilidad total y
  la lista de fallos planificados y espontáneos.
- **C20**: el micrófono vuelve a capturar en **<=180 s** tras reconectarse y
  enumerarse. Las entradas sanas mantienen su heartbeat durante la falla. Los
  reintentos son acotados (backoff del watchdog de 1 s a 10 s, configurado en
  `rooms.stream_watchdog`).

## Reglas de la ventana

- Una sola falla por vez y un solo dispositivo por vez. No resetear hubs
  compartidos, no hacer `unbind`/`authorize=0` por software, no cambiar
  firmware, canal, routing, alimentación ni parámetros del DSP.
- No reiniciar `kza-voice.service` salvo en la falla F4, y sólo con la
  autorización de esa ventana.
- No tocar servicios ni dispositivos de otros proyectos del servidor.
- Las salidas contienen identificadores privados del hogar: se guardan con
  `umask 077` en un directorio privado de la cuenta que las produce y no se
  suben a Git. A la evidencia del roadmap sólo van los números y los
  veredictos.
- Cada falla espontánea durante el soak cuenta. Si aparece el cuelgue de
  isócronos (ceros exactos, ver más abajo), se registra como espontáneo.

## Instrumentos

### 1. Diagnóstico de micrófonos (compilado)

`tools/diagnose_room_microphones.py` ya viaja compilado en la release
(`tools/` está en el artefacto). Lee las habitaciones y su `mic_usb_port` de
`config/settings.yaml` (hoy Living, Hall, Cocina y Escritorio; Baño figura en
`rooms_without_microphone`). Es de solo lectura: no abre dispositivos ni
reinicia servicios, y de la consola de KZA sólo exporta líneas
`[audio-watchdog]`, acotadas por habitación.

```sh
umask 077
cd /home/kza/releases/current
/home/kza/kza/.venv/bin/python -B tools/diagnose_room_microphones.pyc \
  > ~/diagnostico-microfonos-$(date -u +%Y%m%dT%H%M%SZ).json
```

Qué mirar para la Cocina:

- `rooms.cocina.usb_port`: el puerto configurado (hoy `7-5.3`).
- `first.cocina.present` / `second.cocina.present`: el dispositivo está en sysfs.
- `second.cocina.capture_state`: `running` cuando KZA está capturando.
- `capture_progress.cocina.hw_ptr_advanced`: `true` si el puntero ALSA avanzó
  entre las dos muestras (3 s). **No prueba audio real**: el cuelgue de
  isócronos entrega ceros al ritmo correcto (runbook
  `2026-07-25-diagnostico-mic-mudo-isocronos.md`). El heartbeat sí lo detecta.
- `kernel_last_2h`: si `exit_code` no es 0, la cuenta no puede leer el journal
  del kernel; la enumeración se toma entonces del muestreador (abajo).
- `voice_watchdog_log_tail.rooms.cocina`: últimas líneas del watchdog de la
  Cocina, sin que una avalancha de otra habitación las oculte.

### 2. Heartbeat de audio

KZA escribe `data/audio_health.json` del estado del slot cada
`check_interval_s` (2 s), con `wall` (hora de escritura) y por habitación
`age_s` (segundos desde el último audio **no-cero**) y `ever` (si entregó audio
real desde el arranque).

Heartbeat esperado de una entrada sana:

- `ever` es `true` y `age_s` <= `no_frames_timeout_s` (8 s).
- El snapshot está fresco: `t - wall` <= 10 s en cada muestra.
- Ancla de audio real: `wall - age_s`. Sólo avanza cuando llega audio no-cero;
  una reapertura del stream no la mueve.

### 3. Muestreador de la ventana

Desde `kza`, solo lectura, una línea JSON por segundo. `PORT` es el
`mic_usb_port` de la Cocina que informó el diagnóstico.

```sh
umask 077
PORT=7-5.3
STATE=$(readlink -f /home/kza/releases/current)/../state
OUT=~/cocina-ventana-$(date -u +%Y%m%dT%H%M%SZ).jsonl
while sleep 1; do
  printf '{"t":%s,"present":%s,"health":%s}\n' "$(date +%s.%N)" \
    "$(test -e /sys/bus/usb/devices/$PORT && echo true || echo false)" \
    "$(cat "$STATE/data/audio_health.json" 2>/dev/null || echo null)" >> "$OUT"
done
```

Se corta con Ctrl-C al cerrar la ventana. Si el archivo `health` sale `null`,
la línea cuenta como heartbeat ausente para todas las entradas.

## Precondiciones (antes de cada ventana)

1. Candidato congelado instalado; anotar el slot y el commit activos.
2. Diagnóstico: las cuatro habitaciones con `present: true`,
   `capture_state: running` y `hw_ptr_advanced: true`.
3. Heartbeat: las cuatro con `ever: true` y `age_s` <= 8 s durante 5 min
   seguidos en el muestreador.
4. Anotar en el registro privado la hora de inicio y quién manipula el
   hardware.

Si una precondición falla, no se inyecta nada: se registra como fallo
espontáneo y se diagnostica por separado.

## Fallas preregistradas (C20)

Cada una con el muestreador corriendo desde 5 min antes hasta 5 min después de
la recuperación (o del límite de 180 s, lo que ocurra último). Entre fallas,
volver a cumplir las precondiciones.

| Id | Acción física | Qué se espera |
|----|---------------|---------------|
| F1 | Desenchufar el XVF3800 de la Cocina del hub del extensor, esperar 30 s y volver a enchufarlo **en el mismo puerto**. | `present` pasa a `false` y vuelve a `true`. Watchdog: `cocina: device <puerto> ausente — esperando en background`, reintentos `absent, retry in Ns` con N <= 10, y `cocina: recovered`. Recuperación <=180 s. Las demás entradas mantienen el heartbeat. |
| F2 | Igual que F1 pero desenchufado 10 min. | Además de F1: reintentos acotados (como máximo uno cada ~10 s, sin avalancha en la consola ni en el journal del kernel). |
| F3 | Desconectar el extensor Cat5e de la Cocina del lado del servidor, 30 s, y reconectarlo. Arrastra el adaptador BT de la Cocina (cuelga del mismo hub). **Pendiente de autorización (pregunta abierta 2).** | Como F1 para el micrófono; la presencia BLE de la Cocina queda fuera de este criterio y se anota aparte. |
| F4 | Con la Cocina desenchufada, reiniciar `kza-voice.service` en la ventana autorizada; esperar el arranque y enchufar la Cocina. | El servicio arranca con las otras tres entradas sanas; la Cocina se recupera <=180 s desde su enumeración. Usa la gracia de primer frame (`first_frame_grace_s`, 180 s). |

El cuelgue de isócronos (el mic enumera, responde al DSP y entrega ceros) no se
puede inyectar a voluntad. Si aparece: `hw_ptr_advanced: true` con `age_s` de la
Cocina creciendo por encima de 8 s. Se sale sólo con un replug físico (F1); se
registra como fallo espontáneo y la recuperación se mide desde ese replug.

## Cálculo (fijado antes de la ventana)

Sobre el `.jsonl` de cada falla:

- **t0** (reconexión y enumeración): la primera muestra con `present: true`
  después del primer `present: false`. Si el journal del kernel está
  disponible, sirve como contraste la línea `usb <puerto>: new high-speed USB
  device`; manda el muestreador.
- **t1** (vuelve a capturar): la primera muestra cuyo ancla de la Cocina
  (`health.wall - health.rooms.cocina.age_s`) es posterior a t0, con `ever:
  true`.
- **Recuperación** = ancla de t1 − t0. Pasa si es <=180 s. Si no hay t1 en 180 s,
  la falla no pasa (no se extiende la espera para el veredicto; se sigue
  observando sólo para diagnóstico).
- **Entradas sanas**: entre el primer `present: false` y t1 (o t0 + 180 s),
  cada muestra cumple, para cada habitación distinta de la Cocina, `ever: true`
  y `age_s` <= 8 s, y `t - wall` <= 10 s. Una sola muestra fuera es un fallo de
  C20 para esa entrada.

## Soak de 72 h (C19)

- Bloque fijo de >=72 h con el candidato congelado, sin inyecciones adentro,
  dentro de los siete días del criterio. Las fallas F1–F4 van antes o después.
- Disponibilidad por entrada = segundos útiles / segundos del bloque. Un
  segundo es útil si la muestra de ese segundo cumple el heartbeat esperado de
  esa entrada (pregunta abierta 1). Una muestra faltante del muestreador cuenta
  como no útil para todas las entradas.
- Se reportan: disponibilidad de la Cocina, de cada otra entrada y total;
  lista de fallos espontáneos con hora y duración; pausas autenticadas
  intencionales identificadas por la política (no se descuentan a mano).
- 99,9 % de 72 h son 259 s no útiles como máximo por entrada.

## Evidencia

Al roadmap (`data/autonomous-roadmap-20261102/t025/`, sin identificadores
privados): commit y slot del candidato, inicio y fin de cada ventana, por falla
t0, t1, recuperación y peor `age_s` de cada entrada sana, y por el soak la
disponibilidad de cada entrada con su lista de fallos. Los `.jsonl` y JSON del
diagnóstico quedan en el directorio privado de quien los tomó.

## Preguntas abiertas (para el usuario, antes de P7)

1. **Segundo útil del soak.** Recomendado: útil = heartbeat esperado (`ever`,
   `age_s` <= 8 s y snapshot fresco). Alternativa: exigir además que el
   diagnóstico muestre captura `running`, que agrega otra fuente sin mejorar la
   señal.
2. **F3 (extensor completo).** Recomendado: autorizarla en una ventana propia
   después de F1 y F2, porque es el modo de falla real del extensor; arrastra el
   BT de la Cocina, que se anota aparte.
3. **Muestreo de 72 h.** El muestreador de arriba es un bucle interactivo bajo
   `kza`. Recomendado: correrlo con `systemd-run --user` como unidad transitoria
   de solo lectura, sin instalar una unit nueva; alternativa: una sesión
   `tmux` de `kza`. Ninguna de las dos se usó todavía en producción.
