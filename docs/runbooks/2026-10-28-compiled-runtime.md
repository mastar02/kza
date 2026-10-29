# Runtime compilado y actualizaciones de código

El servicio y el dashboard ejecutan `release_20261029T174830709997Z_63ea8d3` bajo
`kza`. Los artefactos se construyen y prueban como `yo`; no se desarrolla, compila
ni instalan dependencias en producción. La instalación original del 28 y la
actualización del 29 fueron solicitadas por el usuario antes de completar el
roadmap.

## Actualización de orden de voces — 2026-10-29

La release del 29 modifica tres archivos de la aplicación y conserva los mismos
recursos privados canónicos. El nuevo `state` contiene enlaces a configuración,
datos, entorno, modelos y logs del slot anterior. Esto mantiene los localizadores
absolutos que ya registra la autoridad de retención. No eliminar ese estado al
retirar versiones de código.

La publicación verificó 100 filas de dos páginas por la API autenticada y el
hash del frontend servido por HTTPS. Hall y Escritorio entregan cuadros; Hall
conserva `PP_MIN_NN=0.51` en RAM y el resto del perfil estático previo. HA sigue
en `auth_error`, Living espera cuadros y Cocina está ausente. Estos límites
siguen abiertos y no son una aceptación de salud general.

La herramienta se construyó y probó bajo `yo`, y el operador ejecutó únicamente
su bytecode. Guarda el registro privado en
`/home/kza/releases/deployments/voice-order-63ea8d3.json` y diez respaldos SQLite
consistentes en
`/home/kza/releases/backups/release_20261029T174830709997Z_63ea8d3/`.
La actualización modifica sólo `WorkingDirectory` en `release.conf`, selecciona
el runtime nuevo y reinicia `kza-voice`; no reinicia el proxy ni cambia modelos,
credenciales o dependencias.

Para revertir **esta actualización de código**, desde `kza`, sin sudo:

```sh
/home/kza/kza/.venv/bin/python -B /tmp/kza-publish-voice-order-20261029-1xge67qw/publish_voice_order.pyc --rollback
```

La herramienta restaura el selector y el drop-in anteriores, reinicia voz y
repone el perfil de Hall. Mantiene los datos actuales: no copia un respaldo viejo
sobre actividad posterior. El bundle temporal puede desaparecer al limpiar
`/tmp`; su fuente está en `tools/publish_voice_order.py` y sus hashes en la
[evidencia de publicación](../research/2026-10-29-voice-order-publication.json).
Si hay que reconstruirlo, hacerlo bajo `yo` y transferir sólo el bytecode y sus
recursos verificados. El rollback histórico al checkout original, al final de
este documento, corresponde a la instalación del 28.

## Acceso y comprobación

Abrir **https://192.168.1.2:9540/obs/** desde la red de casa. El certificado es
local y autofirmado; su SHA-256 es
`b05e58f5c3b577fa22393de9e3a67866bf3c972a70ac9ee7d44a0190083deaa0`.
El login usa el `KZA_DASHBOARD_TOKEN` existente en
`/home/kza/secrets/.env`. Mantener ese valor fuera del chat y de Git.

Desde una sesión de `kza`:

```sh
systemctl --user status kza-voice.service kza-dashboard-proxy.service
systemctl --user status kza-console-rotate.timer
readlink -f /home/kza/releases/current
```

El backend escucha sólo en `127.0.0.1:9500`; el proxy del proyecto expone TLS en
`192.168.1.2:9540`. La shell y los assets son públicos. Las APIs privadas y el
canal en vivo requieren sesión o Bearer; `/api/health` sin credenciales devuelve
401. No usar un HTTP 200 de la shell como prueba de salud de HA o de audio.

El login real, dashboard renderizado y WebSocket pasaron en Chromium sin errores
JavaScript. Una observación HTTP de dos minutos tuvo 13 muestras correctas.
La salud autenticada sigue en `unhealthy`: HA devuelve 403 desde el servidor
KZA y tres de las cuatro entradas de audio no entregan frames. El arranque local
continúa y HA conserva su reconexión acotada existente. No se verificaron
órdenes habladas, reproducción física ni efectos sobre dispositivos HA.

## Archivos y logs

- Slot activo: `/home/kza/releases/slots/release_20261029T174830709997Z_63ea8d3/`.
- `runtime`: bytecode y assets verificados; `release.json` contiene el commit y
  los hashes. Los enlaces `config`, `data` y `.env` apuntan al estado del slot.
- `state`: contenedor de enlaces a los recursos privados canónicos del slot del
  28. El contenedor es el `WorkingDirectory`; los datos siguen en sus rutas
  originales y `models` conserva el directorio existente.
- El drop-in `~/.config/systemd/user/kza-voice.service.d/release.conf` selecciona
  el runtime con `PYTHONPATH`, `CONFIG_PATH`, `-B` y `-P`. Las credenciales siguen
  entrando por los EnvironmentFiles originales.
- La consola va a `state/logs/console.log`. El timer ejecuta
  `/home/kza/releases/console-logrotate.conf` cada hora, con 24 rotaciones,
  `maxage 1` y `copytruncate`. Se comprobó una ejecución correcta de logrotate.
- El proxy tiene configuración y logs propios bajo
  `/home/kza/releases/dashboard-proxy/`. No cambia el NGINX compartido.

La salida de consola a archivo es una contingencia: el journal global se
atascó y bloqueó stdout de KZA. El primer stop agotó los 90 segundos de systemd
y terminó por SIGKILL. Se comprobaron las diez bases SQLite del estado antes
de volver a iniciar. El usuario reinició journald, pero luego volvió a observarse
CPU alta; el incidente global sigue sin diagnóstico cerrado. No hay evidencia
de apagado gradual aceptado ni de borrado global por este procedimiento.

## Lectura de consola desde `yo`

El 28 de octubre el usuario aplicó una ACL desde su sesión `kza`. `yo` tiene
lectura de `console.log` y sólo tránsito por los directorios necesarios. Se
comprobó una lectura real de 4096 bytes, sin imprimir contenido privado, y se
verificó que no puede escribir en el archivo ni listar el directorio de logs.
Tampoco puede leer el `.env` del estado, `/home/kza/secrets/.env`, la
configuración privada ni listar el directorio de datos.

Desde `yo`, para seguir la consola de la release actual:

```sh
runtime_path="$(readlink -f /home/kza/releases/current)"
tail -n 100 -F "$(dirname "$runtime_path")/state/logs/console.log"
```

No usar `current/../state` desde `yo`: ese recorrido exige atravesar el
directorio `runtime`, que conserva sus permisos privados. La ACL no habilita
otros logs, archivos rotados, alertas por API ni operaciones de systemd.
`copytruncate` conserva el inode de la consola y su permiso de lectura. Una
release futura necesita reaplicar el permiso sobre su nuevo slot.

Para reaplicarlo, desde una sesión de `kza`, respaldar primero las ACL de los
directorios afectados y del archivo con `getfacl -p`, y luego ejecutar:

```sh
slot="$(dirname "$(readlink -f /home/kza/releases/current)")"
setfacl -m u:yo:r-- "$slot/state/logs/console.log"
setfacl -m u:yo:--x /home/kza/releases /home/kza/releases/slots \
  "$slot" "$slot/state" "$slot/state/logs"
```

El respaldo de esta operación quedó en
`/home/kza/.local/state/kza-log-read/20261028T202838023713Z/before.acl`.
`kza` puede restaurarlo con `setfacl --restore=RUTA_DEL_RESPALDO`. El permiso no
requirió reiniciar el servicio ni crear otra clave SSH.

## Rollback conservando ambos estados

El respaldo original es
`/home/kza/releases/backups/release_20261028T144002Z_2c5adb4/`.
Incluye unit, drop-ins, configuración, datos, entorno privado y manifiesto de
hashes. El sufijo identifica la operación de respaldo; no identifica el código
que ejecutaba la producción anterior. Sus ocho SQLite pasaron `quick_check`.
La configuración y los datos originales conservan los hashes del respaldo.

Desde una sesión de `kza`:

1. Detener `kza-voice.service`. Deshabilitar y detener
   `kza-dashboard-proxy.service` y `kza-console-rotate.timer`.
2. Apartar únicamente el nuevo drop-in `release.conf`, conservándolo junto al
   respaldo para diagnóstico. Mantener los drop-ins originales de credenciales.
3. Ejecutar `systemctl --user daemon-reload` y comprobar que ExecStart y
   WorkingDirectory vuelven a las rutas originales de `/home/kza/app`.
4. Arrancar la versión anterior sólo si se decide restaurar su ejecución. Antes
   de esta instalación la unit estaba inactiva, aunque habilitada.
5. Conservar el slot nuevo y su estado. No copiar sus datos encima del original:
   los cambios posteriores al corte necesitan una reconciliación explícita.

No borrar `/home/kza/kza`, `/home/kza/app`, el venv ni los modelos: además del
rollback, el índice existente todavía depende del checkout anterior. La clave
SSH temporal usada para esta operación se retira al terminar; este runbook
presupone una sesión administrativa propia del operador.
