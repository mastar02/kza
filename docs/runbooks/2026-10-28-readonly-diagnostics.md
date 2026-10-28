# Diagnóstico de KZA desde `yo`

El usuario instaló el 28 de octubre una herramienta de consulta fija bajo
`kza`. Permite a `yo` observar la release y sus dependencias sin una shell en la
cuenta de ejecución. Es independiente del pipeline: no reinició ni sustituyó
`kza-voice.service`, que continúa en `b0866e8`.

## Consulta

Desde `yo`, ejecutar y guardar el resultado sólo en un directorio privado:

```sh
umask 077
ssh -F /dev/null -i /home/yo/.ssh/kza_diagnostics_ed25519 \
  -o IdentityAgent=none -o IdentitiesOnly=yes -o BatchMode=yes \
  -o ConnectTimeout=5 -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/home/yo/.ssh/kza_diagnostics_known_hosts \
  kza@127.0.0.1 snapshot
```

La clave está restringida a conexiones desde `127.0.0.1`, sin PTY ni forwarding.
Su comando forzado ejecuta siempre el mismo snapshot, ignorando el comando y
los argumentos que envíe el cliente. Se comprobó enviando un comando `echo` y
recibiendo únicamente el objeto de diagnóstico.

La salida incluye estado de las units, E/S y esperas del proceso, configuración
seleccionada de captura, heartbeat, APIs autenticadas de salud y micrófonos,
resumen de alertas y los últimos mensajes del servicio journald. No devuelve
los archivos de entorno, las credenciales, el entorno del proceso, audio ni
transcripciones. Los resultados contienen identificadores privados del hogar;
no subirlos a Git.

El snapshot es una consulta puntual: comparar dos observaciones para medir
tasas y detectar contadores reiniciados. No confundir bytes lógicos de lectura
con lecturas físicas de disco. Las métricas de cgroup pueden reiniciarse al
recrearse una unit; no calcular tasas con deltas negativos.

## Artefacto y límites

Fuente: `tools/runtime_diagnostics.py`, commit
`ecf4a6ed8cded2dbeafa804bfb3170fce957b138`.
Se construyó bajo `yo` como bytecode CPython 3.13 y se instaló únicamente
`/home/kza/releases/diagnostics/ecf4a6e/runtime_diagnostics.pyc`.
SHA-256: `c9cb1d1431406bb896661205991ab72128fee72d0c424efef8ab2ac050d9cca4`.
Pasaron seis casos de la herramienta desde fuentes y los mismos seis desde el
artefacto compilado. Ruff pasó; Bandit registró un hallazgo bajo B404 y ninguno
medio o alto. No se instaló ninguna dependencia en producción.

La clave no permite cambiar la herramienta instalada, consultar archivos o
endpoints arbitrarios, controlar dispositivos, reiniciar servicios ni obtener
privilegios de root. Las nuevas versiones requieren que el operador instale
otro artefacto verificado y actualice el comando forzado. No concede acceso
general al journal del kernel o a los datos de otros proyectos. Los errores de
consulta se reportan como tipo de error, sin publicar su mensaje privado.

Para revocar el acceso, `kza` debe retirar de `~/.ssh/authorized_keys` la entrada
con comentario `yo-kza-diagnostics-readonly-20261028`, conservando las demás.
Después verificar que esa clave ya no autentica y retirar su material local de
`yo`. El instalador guardó las entradas previas bajo
`~/.local/state/kza-diagnostics-access/`. No restaurar un respaldo antiguo a
ciegas si después se agregaron otras claves.

## Diagnóstico del 28 de octubre

La investigación posterior identificó la avalancha USB que atascaba journald y
realizó una prueba de aislamiento. Ver el
[diagnóstico específico de journald](2026-10-28-journald-usb-flood.md); la lista
siguiente conserva lo conocido en la primera observación.

La [evidencia resumida](../research/2026-10-28-runtime-diagnostic.json) separa
hechos observados de causas todavía sin demostrar:

- La muestra de 120 segundos tuvo PSI de E/S máximo de 0,03 % en el host y
  0 % en el cgroup de voz. El proceso escribió 532.480 bytes físicos en otra
  ventana de 42,43 segundos. No atribuye ni descarta picos históricos.
- Journald consumió aproximadamente un núcleo en los intervalos válidos y su
  contador de reinicios pasó de 80 a 82. Su journal registra mensajes repetidos
  de pérdida de mensajes del kernel y un timeout de watchdog con SIGABRT.
  Falta identificar el origen del problema del journal/kernel; no se demuestra
  que KZA lo haya causado.
- Dos capturas de KZA no avanzaron sus punteros ALSA en 61 observaciones. Otra
  entregó callbacks sin señal no-cero en el canal seleccionado. Una cuarta
  siguió recibiendo audio. La presencia USB y una consulta DSP exitosa no prueban
  la captura. No se cambió firmware, canal, routing, alimentación ni parámetros.
- El heartbeat de audio quedó 240 segundos viejo mientras la API mostraba una
  captura sana con callbacks recientes. El poller interpretaría las cuatro
  entradas como caídas. El código publica desde la misma coroutine que espera
  `_recover_streams`; un probe aislado reproduce la falta de progreso. Cuatro
  ciclos reales tardaron entre 318,348 y 318,576 segundos hasta la primera
  reapertura. Es un defecto de diagnóstico pendiente de corregir en una release.
- HA sigue devolviendo 403 desde KZA, incluso en `/` sin credenciales. La API
  autenticada de salud reporta `auth_error`. No se ha distinguido entre bloqueo
  de origen, filtro o configuración de autenticación; no se renovaron tokens
  ni se intentó eludir el rechazo.
- La API de alertas devolvió una lista vacía con fuente `real`, mientras salud
  estaba en `unhealthy`. Esa lista no basta para descartar fallos del runtime.

La próxima corrección de código debe mantener el heartbeat independiente de
recuperaciones lentas, conservar el ownership del driver y evitar reaperturas
solapadas. Una recuperación no puede declarar sanos los streams por el solo
hecho de abrirlos ni reiniciar el contador de sordera. Los fallos físicos
necesitan una prueba controlada de un dispositivo por vez; no resetear hubs
compartidos ni manipular servicios de otros proyectos.
