# Reinicios de journald por avalancha de captura USB

El 28 de octubre se identificó una avalancha del driver xHCI asociada a una
captura USB de KZA. El kernel emitía miles de avisos por segundo; journald
perdía mensajes, consumía CPU y agotaba timeouts de arranque o de watchdog.
Retirar únicamente el micrófono identificado detuvo la avalancha y redujo la
CPU de journald. La prueba de aislamiento y su alcance están en la
[evidencia numérica](../research/2026-10-28-journald-usb-flood.json).

## Evidencia y atribución

Una consulta de kernel contenía 1000 mensajes idénticos en 0,137497 segundos
de tiempo de origen: unos 7266 mensajes por segundo. Rsyslog también registró
series de 40.007 repeticiones en cinco segundos, y 120.014 callbacks de captura
suprimidos en ese período. El texto repetido era un `buffer overrun event`
del controlador xHCI, sobre un mismo slot y endpoint de captura.

La correspondencia entre slot y dispositivo se obtuvo con una lectura de
debugfs realizada por el operador como root. Se cruzó con la identidad USB,
la configuración de captura y la API real de micrófonos de KZA. El puerto,
habitación, serial y los logs originales quedan en el directorio privado
`data/journal-diagnostic-20261028/`, fuera de Git.

El journal tenía un retraso de aproximadamente 38 segundos respecto del
tiempo de origen del kernel en la muestra. Syslog registra timeouts de
watchdog de tres minutos, timeouts de arranque y reinicios automáticos. También
hay avisos de journals cerrados incorrectamente o corruptos y errores de
rotación `Device or resource busy`; no se borraron archivos para intentar
resolverlos.

La prueba consistió en desconectar físicamente sólo el micrófono involucrado.
Se mantuvieron el hub y el controlador, compartidos con otros dispositivos y
un disco externo. No se reinició journald, KZA ni otro servicio durante esa
prueba. Las métricas se tomaron antes y después, incluyendo presencia USB,
CPU del cgroup, contador de secuencia del kernel y estado/reinicios de la unit.

Tras el aislamiento se observaron 96 muestras durante 190,20 segundos, además
de la transición inicial. Journald mantuvo el mismo PID y su contador en 88;
su CPU media fue 0,116 % de un núcleo y el cursor del kernel avanzó unas 0,20
posiciones por segundo. El dashboard siguió respondiendo, KZA conservó su PID
y el disco externo permaneció conectado y montado. La ventana cubre el timeout
de watchdog de tres minutos; no equivale a una aceptación prolongada ni a la
reparación del dispositivo.

Esto atribuye el disparador a la ruta de captura USB identificada. No determina
todavía si el origen es el micrófono, firmware, cable/extensor, alimentación,
driver o su interacción con la apertura de captura. Tampoco atribuye todos los
picos históricos de PSI de E/S de otros proyectos a esta incidencia.

## Diagnóstico sin ampliar privilegios

`yo` puede consultar el estado de la unit y métricas públicas de cgroup, pero
no leer el journal global ni `/dev/kmsg`: el kernel rechaza esa lectura aunque
los bits de modo del dispositivo parezcan permitirla. No cambiar
`kernel.dmesg_restrict` ni agregar permisos generales para obtener una muestra.

`kza` pertenece a `adm` y puede hacer una colección acotada de kernel/syslog.
La colección de esta sesión leyó cuatro MiB de cada archivo, filtró syslog a
journald y exportó sólo el diagnóstico a
`/home/kza/kza-journal-diagnostic.json`, con ACL de lectura para `yo`.
El script operativo no cambió servicios y no necesitó sudo.

Para mapear el slot del aviso hace falta una lectura de root. Sustituir el
controlador por el que aparece en el error:

```sh
sudo find /sys/kernel/debug/usb/xhci/CONTROLADOR/devices \
  -maxdepth 2 -name name -print -exec cat {} \;
```

El archivo `name` de cada slot devuelve el puerto USB físico. No confundir el
número de slot xHCI con la dirección USB mostrada por `lsusb`; son espacios de
identificación distintos.

## Contención y siguiente prueba

El dispositivo identificado queda desconectado como contención. Reabrir el
stream, reiniciar journald o modificar su límite de almacenamiento no corrige
por sí solo el origen de una avalancha de errores USB.

La siguiente prueba debe usar un único micrófono, un cable/conexión controlados
y observación simultánea de callbacks, señal útil y avisos del kernel. Si se
reconecta el dispositivo, vigilar la reaparición del error; no restablecerlo sin
esa observación. Cambiar una variable por vez permite separar alimentación,
extensor, firmware y driver. No resetear el controlador ni el hub compartidos.
Los cambios de firmware o kernel requieren una intervención posterior y su
validación específica; no se hicieron como parte de este diagnóstico.

El defecto ya reproducido del heartbeat de KZA permanece pendiente: una
recuperación lenta todavía puede suspender la publicación aunque otra captura
siga sana. Ese defecto es distinto de la avalancha que reiniciaba journald.

## Fuentes técnicas

- [Driver xHCI de Linux 6.8](https://github.com/torvalds/linux/blob/v6.8/drivers/usb/host/xhci-ring.c).
- [Mapeo de slots en debugfs](https://github.com/torvalds/linux/blob/v6.8/drivers/usb/host/xhci-debugfs.c).
- [Lectura de mensajes del kernel en systemd 255](https://github.com/systemd/systemd/blob/v255/src/journal/journald-kmsg.c).

Las fuentes explican las interfaces. La atribución al dispositivo de este
servidor procede de las mediciones y de la prueba de aislamiento, no de un
reporte externo con síntomas parecidos.
