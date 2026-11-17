# Aceptación preregistrada de la Cocina (C19/C20)

Estado: **preregistrado, no ejecutado**. Lo escribió T025 (P2) el 2026-11-16 sin
tocar el servidor de producción. La ejecución física (micrófono XVF3800 de la
Cocina, su extensor Cat5e, el soak de 72 h y la inyección de fallos) queda para
P7, con un candidato congelado. Este guion se fija **antes** de mirar resultados:
los umbrales, las fallas a inyectar y el modo de cálculo no se cambian después.
Un cambio exige una versión nueva del guion, fechada, antes de la ventana.

Revisión 1 (2026-11-16, ronda de revisión 1, antes de cualquier ventana): el
muestreador se alinea al segundo de reloj y el cálculo de C19 fija cómo se
asigna cada muestra a un segundo y cuánto hueco del muestreador se tolera
(antes, la deriva del bucle `sleep 1` hacía fallar C19 por sí sola). Se agregan
la pausa autenticada intencional (pregunta abierta 4), el estado conocido de
Living y Cocina, y el alcance de C19 que este guion no mide.

Revisión 2 (2026-11-16, antes de cualquier ventana): el usuario respondió las
preguntas abiertas (decisión `t025_p4d0_questions_20261116` en
`data/autonomous-roadmap-20261102/t040-t041/user-decisions-20261110.json`).
Segundo útil = heartbeat sano; silencio global prohibido dentro del bloque de
72 h; F3 autorizada en una ventana propia; la captura `closed` de la Cocina se
diagnostica con el usuario, como `kza`, antes de congelar el candidato; el
muestreador corre como unidad transitoria de `systemd-run --user`; la
disponibilidad de KZA como servicio va a P6/P7 con un instrumento propio. El
cálculo de C19 no cambia: las respuestas eligen las opciones que ya suponía.
Ver «Preguntas respondidas».

Revisión 3 (2026-11-17, antes de cualquier ventana): la Cocina volvió a
capturar al reiniciar `kza-voice` y se encontró la causa raíz de su captura
`closed` (ver «Estado conocido antes de P7»). T025-U2 agrega el refresco
automático del catálogo de PortAudio con un corte breve de captura en todas
las habitaciones (decisión del usuario «Corte breve automático»,
`home_return_20261117`). Se agregan las líneas de consola que lo muestran y
cómo cuenta ese corte en F1–F4. Los criterios, los umbrales y el cálculo no
cambian.

Alcance: la entrada de audio de la Cocina y el efecto sobre las demás entradas.
La parte de C20 sobre HA/WebSocket (recuperación en <=60 s) no está en este
guion. De C19, este guion mide sólo la disponibilidad de cada entrada de audio
en el bloque físico. **No mide** la disponibilidad de KZA como servicio ni el
soak local de 72 h: el soak local es de P6 (Candidato) y la disponibilidad de
KZA se mide en P6/P7 con un instrumento propio, todavía sin escribir (ver
«Fuera de este guion»). La frescura del snapshot (`t - wall` <= 10 s) es aquí
una condición del heartbeat de cada entrada, no una medida de la
disponibilidad de KZA.

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

## Estado conocido antes de P7

- `docs/architecture/DEPLOYMENT.md:47-48` (instalación del 2026-10-29): «La
  observación posterior confirma captura en Hall y Escritorio; Living y Cocina
  siguen sin captura útil».
- `docs/architecture/DEPLOYMENT.md:107-115`: una avalancha de captura USB tumbó
  journald; «retirar sólo el micrófono involucrado» lo recuperó y «el defecto
  de la ruta USB sigue pendiente». El documento no dice qué micrófono fue.
- Observación de T025 (2026-11-16, como `yo`, diagnóstico compilado, sin leer
  nada bajo `/home/kza`): Living, Hall y Escritorio presentes con captura
  `running` y `hw_ptr` avanzando; Cocina (`7-5.3`) presente con captura
  `closed`.
- Conciliación de Living: las dos observaciones no se contradicen del todo,
  pero tampoco se pueden conciliar con lo que se midió. `hw_ptr` avanzando
  **no prueba audio real** (el cuelgue de isócronos entrega ceros al ritmo
  correcto); «captura útil» en DEPLOYMENT.md se refiere a audio real. O Living
  se recuperó después del 2026-10-29, o sigue entregando ceros. Lo decide sólo
  el heartbeat (`age_s`, `ever`) de `data/audio_health.json`, que vive en el
  `data/` `0700` de `kza` y `yo` no lee. Se verifica como `kza` en la
  precondición 3 de la primera ventana; hasta entonces el estado de Living
  queda **sin confirmar**.
- Restauración (2026-11-17, con el usuario en casa): con su autorización
  («Sí, reiniciar ahora») se reinició `kza-voice` como `kza` a las 00:00:19.
  El heartbeat de la Cocina pasó a `ever: true` a las 00:00:57 y las cuatro
  habitaciones quedaron capturando. El reinicio no cuenta como falla
  inyectada ni como ventana de este guion.
- Causa raíz de la captura `closed`: el XVF3800 de la Cocina (puerto `7-5.3`)
  enumeró como tarjeta ALSA 6 el 2026-11-15 a las 16:02, cuando `kza-voice` ya
  corría desde el 2026-11-02. PortAudio arma su lista de dispositivos una sola
  vez, al inicializarse (`Pa_Initialize`), y no la vuelve a leer. Por eso
  `resolve_mic_usb_port` encontraba la tarjeta en sysfs pero no en PortAudio,
  y la consola repitió cada 10 s durante días `Mic USB port 7-5.3 → ALSA card
  6, pero PortAudio no lista esa card`. `MultiRoomAudioLoop` sólo
  reinicializaba PortAudio cuando ninguna otra habitación estaba sana; con
  tres sanas, el micrófono tardío no se veía hasta reiniciar el servicio.
- Arreglo (T025-U2, desarrollado, **sin release**): cuando la tarjeta de un
  micrófono configurado está en sysfs y PortAudio no la lista, la espera de
  reapertura de esa habitación pide un refresco del catálogo por la ruta de
  recuperación global que ya existía (`InputGroup.catalog_call`, con la época
  de recuperación): cierra la captura de **todas** las habitaciones,
  reinicializa PortAudio y las reabre todas. Límites: como mucho un refresco
  cada `rooms.stream_watchdog.catalog_refresh_min_interval_s` (300 s en
  `config/settings.yaml`; 0 lo desactiva); nunca mientras se captura o
  despacha un comando (`CommandTurns`), con una conversación activa
  (`FollowUpMode`) o con una respuesta de voz en curso (un reinicio de
  PortAudio también cortaría la salida). Esas condiciones se vuelven a mirar
  al empezar el corte y justo antes de reinicializar; si aparecen después de
  cerrar las entradas, no se reinicializa y sólo se reabre. Si la tarjeta
  tampoco está en sysfs (micrófono desenchufado) no hay refresco. Hasta que
  una release lo instale, un micrófono que enumera después del arranque sigue
  necesitando reiniciar `kza-voice`.
- Límite conocido del refresco (revisión T025-U2-RV2-1): si `Pa_Initialize`
  falla durante el refresco, PortAudio queda sin inicializar y **ninguna**
  habitación puede abrir, tampoco las que estaban sanas. Desde RV2-1 el
  reinicio separa `terminate` de `initialize` (antes, tras un fallo así,
  `terminate` fallaba en cada intento y `initialize` no volvía a correr: la
  casa quedaba sorda hasta reiniciar), y el refresco siguiente no espera los
  300 s sino un backoff corto: 10 s (techo del backoff de reapertura),
  duplicándose por fallo seguido hasta 300 s. Mientras dure, la casa está
  sorda; con el refresco desactivado (intervalo 0) sigue requiriendo
  reiniciar `kza-voice`. En consola: `PortAudio reinit failed`, luego
  `PortAudio is not initialized after the catalog reset; ... retrying the
  refresh in N s` y, en el reintento, `PortAudio terminate failed: ...
  PortAudio not initialized` antes de reabrir.
- Tarjeta que PortAudio no lista ni después del refresco (revisión
  T025-U2-RV2-3: dispositivo ocupado, interfaz de captura rota): antes cada
  300 s había un corte de toda la casa, sin fin (~288 por día). Ahora el
  segundo refresco espera 300 s y cada refresco inútil seguido duplica la
  espera (600, 1200, 2400 s) hasta
  `rooms.stream_watchdog.catalog_refresh_max_interval_s` (3600 s); vuelve a
  300 s si la tarjeta sale de sysfs o cambia de número. Al segundo refresco
  inútil se registra una sola advertencia por racha: `<room>: PortAudio still
  does not list ALSA card N after 2 catalog refreshes; backing off the next
  refresh to 600s, doubling up to 3600s`.
- Consecuencia: la precondición 2 (las cuatro habitaciones `running`) y la 3
  (las cuatro con heartbeat) se vuelven a comprobar en cada ventana, como
  siempre. La captura `closed` de la Cocina quedó explicada y resuelta por el
  reinicio del 2026-11-17; el heartbeat de Living se confirma en la
  precondición 3 de la primera ventana. El candidato que se congele para P7
  debe incluir T025-U2: sin él, F4 (y F1–F3 si la tarjeta vuelve con otro
  número) no puede recuperarse sin reinicio.

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
- Dentro del bloque de C19 está **prohibido** activar el silencio global de
  captura (pregunta 4). Si ocurre igual, se registra con su evento de política
  y cuenta como dice «Pausa autenticada intencional».

## Instrumentos

### 1. Diagnóstico de micrófonos (compilado)

`tools/diagnose_room_microphones.py` ya viaja compilado en la release
(`tools/` está en el artefacto). Lee las habitaciones y su `mic_usb_port` de
`config/settings.yaml` (hoy Living, Hall, Cocina y Escritorio; Baño figura en
`rooms_without_microphone`). Es de solo lectura: no abre dispositivos ni
reinicia servicios, y de la consola de KZA sólo exporta líneas
`[audio-watchdog]`, acotadas por habitación. Atribuye a cada habitación sólo
las tarjetas de sonido de las interfaces propias de su puerto
(`<puerto>:*/sound/cardN`), la misma regla con la que KZA asocia el micrófono
(`src/rooms/room_context.py`, `usb_port_to_alsa_card`): un hub nunca se queda
con la captura de un dispositivo que cuelga de él. Toma una habitación como
toda entrada de `rooms` con `mic_usb_port`; no mira `rooms.enabled`.

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

Mientras la captura está en silencio global (pausa autenticada, ver
«Pausa autenticada intencional»), KZA no publica el heartbeat: el snapshot
deja de estar fresco y la muestra no cumple para ninguna entrada.

### 3. Muestreador de la ventana

Bajo `kza`, solo lectura, una línea JSON por **segundo de reloj**. `PORT`
(primer argumento del script, hoy `7-5.3`) es el `mic_usb_port` de la Cocina
que informó el diagnóstico. Requiere `bash`
(aritmética `10#`) y `date` de GNU (`%N`).

Corre como unidad **transitoria** de `systemd-run --user` (pregunta 3): no se
instala ninguna unit y `--collect` la descarta al detenerla. `kza` tiene
linger (`DEPLOYMENT.md:105`), así que la unidad sigue viva al cerrar la sesión
SSH. El bucle va en un script del directorio privado de salida y no en la
línea de comando, porque systemd puede expandir `$VAR` en los argumentos de
la unidad. Este comando todavía no se ejecutó en el servidor: se comprueba en
la precondición 3 de la primera ventana (unidad `active` y una línea por
segundo en el `.jsonl`).

Cada vuelta duerme hasta el próximo segundo entero del reloj y recién ahí
toma la muestra, así el error de una vuelta no se acumula. El bucle de la
versión anterior (`while sleep 1`) sumaba el costo de cada vuelta: con unos
3 ms por vuelta son ~950 s sin muestra en 72 h, más que los 259 s que admite
C19 (medición del revisor; la prueba de 30 min de esta revisión está en la
evidencia de T025).

```bash
umask 077
mkdir -p ~/cocina-ventanas
TS=$(date -u +%Y%m%dT%H%M%SZ)               # anotarlo en el registro privado
cat > ~/cocina-ventanas/muestreador-$TS.sh <<'EOF'
PORT=$1
OUT=$2
STATE=$(readlink -f /home/kza/releases/current)/../state
while :; do
  rem=$(( 1000000000 - 10#$(date +%N) ))      # ns hasta el próximo segundo
  sleep "$(( rem / 1000000000 )).$(printf %09d $(( rem % 1000000000 )))"
  printf '{"t":%s,"present":%s,"health":%s}\n' "$(date +%s.%N)" \
    "$(test -e /sys/bus/usb/devices/$PORT && echo true || echo false)" \
    "$(cat "$STATE/data/audio_health.json" 2>/dev/null || echo null)" >> "$OUT"
done
EOF
systemd-run --user --unit=kza-cocina-muestreo-$TS --collect -p UMask=0077 \
  /bin/bash ~/cocina-ventanas/muestreador-$TS.sh 7-5.3 \
  ~/cocina-ventanas/ventana-$TS.jsonl
systemctl --user is-active kza-cocina-muestreo-$TS   # -> active
```

Se detiene al cerrar la ventana (para C19, después del fin del bloque; ver
«Soak de 72 h») con `systemctl --user stop kza-cocina-muestreo-$TS`. La unidad
no se reinicia sola: si muere, los segundos sin muestra son huecos del
muestreador y cuentan como tales. El script y el `.jsonl` quedan en
`~/cocina-ventanas/` (`0700`). Si el archivo `health` sale `null`, la línea
cuenta como heartbeat ausente para todas las entradas. El muestreador no
distingue una pausa autenticada de una caída: eso lo da la fuente de política
(abajo).

## Precondiciones (antes de cada ventana)

1. Candidato congelado instalado; anotar el slot y el commit activos.
2. Diagnóstico: las cuatro habitaciones con `present: true`,
   `capture_state: running` y `hw_ptr_advanced: true`.
3. Heartbeat: las cuatro con `ever: true` y `age_s` <= 8 s durante 5 min
   seguidos en el muestreador.
4. Anotar en el registro privado la hora de inicio y quién manipula el
   hardware.
5. Captura sin silencio global: estado de captura del panel de administración
   (`GET /api/capture/control`, sólo `admin:manage`) con `phase: running`;
   anotar su `revision`.

Si una precondición falla, no se inyecta nada: se registra como fallo
espontáneo y se diagnostica por separado. Hoy fallan la 2 y la 3 por la
Cocina; se resuelven en el diagnóstico conjunto previo al congelamiento del
candidato (ver «Estado conocido antes de P7»).

## Fallas preregistradas (C20)

Cada una con el muestreador corriendo desde 5 min antes hasta 5 min después de
la recuperación (o del límite de 180 s, lo que ocurra último). Entre fallas,
volver a cumplir las precondiciones.

| Id | Acción física | Qué se espera |
|----|---------------|---------------|
| F1 | Desenchufar el XVF3800 de la Cocina del hub del extensor, esperar 30 s y volver a enchufarlo **en el mismo puerto**. | `present` pasa a `false` y vuelve a `true`. Watchdog: `cocina: device <puerto> ausente — esperando en background`, reintentos `absent, retry in Ns` con N <= 10, y `cocina: recovered`. Recuperación <=180 s. Las demás entradas mantienen el heartbeat. |
| F2 | Igual que F1 pero desenchufado 10 min. | Además de F1: reintentos acotados (como máximo uno cada ~10 s, sin avalancha en la consola ni en el journal del kernel). |
| F3 | Desconectar el extensor Cat5e de la Cocina del lado del servidor, 30 s, y reconectarlo. Arrastra el adaptador BT de la Cocina (cuelga del mismo hub). **Autorizada (pregunta 2) en una ventana propia, después de F1 y F2, con el usuario presente.** | Como F1 para el micrófono; la presencia BLE de la Cocina queda fuera de este criterio y se anota aparte. |
| F4 | Con la Cocina desenchufada, reiniciar `kza-voice.service` en la ventana autorizada; esperar el arranque y enchufar la Cocina. | El servicio arranca con las otras tres entradas sanas; la Cocina se recupera <=180 s desde su enumeración. Usa la gracia de primer frame (`first_frame_grace_s`, 180 s). |

Refresco del catálogo (T025-U2). F4 reproduce exactamente el incidente del
2026-11-15: la tarjeta de la Cocina aparece después de `Pa_Initialize`. En
F1–F3 pasa lo mismo si la tarjeta vuelve con un número que PortAudio no
lista. En esos casos se espera, además de lo que dice la tabla:

- `[audio-watchdog] PortAudio does not list ALSA cards {'cocina': N} present
  in sysfs; refreshing the catalog with a brief capture cut in [...]` con
  todas las habitaciones en la lista;
- `stream reopened` para cada habitación, incluida la Cocina, y después su
  primer audio en el heartbeat;
- si coincide con un comando, una conversación o una respuesta, `PortAudio
  catalog refresh deferred (command|conversation|response)` y el refresco
  apenas termina; `deferred (rate_limit)` si hubo otro refresco hace menos de
  300 s;
- si aparece `still does not list ALSA card N after 2 catalog refreshes`, la
  tarjeta no se recupera con refrescos: se anota y se sale con un replug
  físico (F1) o un reinicio autorizado;
- si aparece `PortAudio is not initialized after the catalog reset`, todas
  las entradas quedan sin audio hasta el reintento (backoff corto, ver
  «Estado conocido antes de P7»): se anota como fallo de C20 de todas las
  entradas, con su duración.

El corte breve alcanza a las entradas sanas. No se excluye del cálculo: si
alguna muestra de una entrada sana sale con `age_s` > 8 s, es un fallo de C20
para esa entrada, como cualquier otro. Para no mezclar efectos, durante la
falla no se dan órdenes de voz ni se deja una conversación abierta.

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

- **Bloque**: un inicio `T_ini` (segundo entero, UTC) y una duración `D` >=
  259 200 s fijados y anotados **antes** de empezar; `T_fin = T_ini + D`. Sin
  inyecciones adentro, dentro de los siete días del criterio. Las fallas F1–F4
  van antes o después. El muestreador arranca al menos 5 min antes de `T_ini`
  y se corta al menos 5 min después de `T_fin`.
- **Segundos del bloque**: los enteros `s` con `T_ini <= s < T_fin`; son `D`
  y son el denominador de cada entrada.
- **Muestra → segundo**: una muestra con hora `t` pertenece al segundo
  `floor(t)`.
- **Muestra de un segundo**: para el segundo `s`, la muestra con `floor(t)`
  entre `s - 2` y `s + 2` cuyo `t` está más cerca de `s + 0,5`; si dos quedan
  a igual distancia, la anterior. Con el muestreador alineado es la propia
  muestra de `s`; las vecinas sólo cubren una vuelta perdida. El margen de 2 s
  es el período con que KZA escribe el heartbeat, así que una muestra a 2 s
  describe el mismo snapshot.
- **Segundo útil** para una entrada (pregunta 1: heartbeat sano): su muestra
  cumple el heartbeat esperado de esa entrada, evaluado con el `t` de esa
  muestra (`ever: true`, `age_s` <= 8 s, `t - wall` <= 10 s). No se exige
  además captura `running` del diagnóstico.
- **Hueco del muestreador** (N = 2 s, fijado ahora): un segundo sin ninguna
  muestra con `floor(t)` entre `s - 2` y `s + 2` es no útil **para todas las
  entradas**. En la práctica, una racha de `k` segundos de reloj seguidos sin
  muestra deja `max(0, k - 4)` segundos no útiles. Los huecos se cuentan y se
  informan aparte de los fallos de las entradas, pero entran en la
  disponibilidad.
- **Disponibilidad por entrada** = segundos útiles / `D`. Pasa si es >=
  99,9 %: como máximo 259 s no útiles por entrada en 72 h (`floor(0,001 · D)`
  para otra `D`).
- Se reportan: disponibilidad de la Cocina, de cada otra entrada y total
  (segundos útiles sumados de todas las entradas / (`D` × número de
  entradas)); lista de fallos espontáneos con hora y duración; huecos del
  muestreador; revisión de captura al inicio y al fin del bloque (pausa
  autenticada, abajo). Nada se descuenta a mano.

### Pausa autenticada intencional

C19 pide que la pausa autenticada intencional se identifique por política.

- Hoy la única pausa autenticada es el silencio global de captura (T023): un
  administrador (`admin:manage`) lo pide con `POST /api/capture/control`
  (`src/dashboard/capture_api.py:70-121`) y KZA lo persiste en la fila
  `capture_runtime_control` (`mode`, `revision`) de `data/memory_privacy.db`
  (`src/memory/capture_control.py:175-204`, `:311-347`). Cada cambio sube la
  `revision`.
- Efecto en el heartbeat: el silencio pausa el lazo de audio
  (`pause_admission`, `src/pipeline/multi_room_audio_loop.py:879-892`), que
  cierra las entradas (`begin_pause_inputs`, `:702-719`) y deja de publicar
  `audio_health.json` (`_publish_audio_health` vuelve sin escribir si
  `_auxiliary_allowed()` es falso, `:2010-2011`, `:721-723`). En el
  muestreador se ve como snapshot viejo: segundos no útiles para todas las
  entradas. Que `last_frame_ts` se actualice antes de `room.accepts()`
  (`:2209-2211`) no sostiene el heartbeat durante el silencio global, porque
  las entradas ya están cerradas; esa compuerta corresponde a los límites por
  conversación de cada habitación, que no son una pausa.
- Fuente de política para la evidencia: la `revision` y la `phase` de
  `GET /api/capture/control`, anotadas al inicio y al fin del bloque
  (precondición 5). Si la `revision` cambió, hubo al menos una transición de
  silencio dentro del bloque. El muestreador **no** lee
  `data/memory_privacy.db`: un lector por segundo sobre la base viva del ledger
  puede competir con su escritor.
- Cómo cuenta (pregunta 4): el silencio global está **prohibido** dentro del
  bloque fijo, como las inyecciones. Si ocurre igual, se registra con su
  evento de política (la `revision` de captura de la precondición 5 y la del
  fin del bloque) y sus segundos cuentan como útiles **sólo si el heartbeat lo
  cumple**, sin descuento manual ni exclusión del denominador. Con el código
  actual el heartbeat no se cumple durante el silencio, así que esos segundos
  salen no útiles. El cálculo de arriba no cambia.

## Evidencia

Al roadmap (`data/autonomous-roadmap-20261102/t025/`, sin identificadores
privados): commit y slot del candidato, inicio y fin de cada ventana, por falla
t0, t1, recuperación y peor `age_s` de cada entrada sana, y por el soak `T_ini`,
`D`, la disponibilidad de cada entrada con su lista de fallos, los huecos del
muestreador y la `revision` de captura al inicio y al fin. Los `.jsonl` y JSON
del diagnóstico quedan en el directorio privado de quien los tomó.

## Fuera de este guion

- Disponibilidad de KZA como servicio (C19 dice «de KZA y de cada entrada»):
  asignada por el usuario a P6/P7 con un instrumento propio, todavía sin
  escribir. La frescura del snapshot no la
  reemplaza: sólo muestra que el watchdog de audio escribe.
- Soak local de 72 h (C19): P6 (Candidato).
- Recuperación de HA/WebSocket en <=60 s (C20).

## Preguntas respondidas (2026-11-16)

Respuestas del usuario, decisión `t025_p4d0_questions_20261116`. En los cinco
casos eligió la opción recomendada.

1. **Segundo útil del soak**: heartbeat sano (`ever: true`, `age_s` <= 8 s,
   snapshot <= 10 s). No se agrega la captura `running` del diagnóstico.
2. **F3 (extensor completo)**: sí, en una ventana propia, después de F1 y F2,
   con el usuario presente. El BT de la Cocina que arrastra se anota aparte.
3. **Muestreo de 72 h**: unidad transitoria de solo lectura con
   `systemd-run --user` bajo `kza`, sin unit instalada (ver «Muestreador de la
   ventana»).
4. **Pausa autenticada dentro del bloque**: prohibida dentro del bloque de
   72 h. Si ocurre, se registra con su evento de política y cuenta como útil
   sólo si el heartbeat lo cumple, sin descuento manual.
5. **Captura `closed` de la Cocina**: se diagnostica juntos, como `kza`,
   cuando el usuario vuelva y antes de congelar el candidato, junto con el
   heartbeat de Living. Es una precondición, no está hecha; sin ella ninguna
   ventana de este guion puede empezar.
6. **Disponibilidad de KZA como servicio (C19)**: P6/P7, con un instrumento
   propio (no la frescura del snapshot de audio).
