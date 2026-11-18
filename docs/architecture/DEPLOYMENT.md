# Deploy de KZA — fuente de verdad

> Si otro documento de este repo contradice a este, gana este (y ante dudas de
> contrato compartido, `docs/SERVER_CONVENTIONS.md` / Notion pág. 8).

## Desarrollo y despliegue acordados — 2026-10-25

En el entorno Linux, el desarrollo, los tests, la construcción y Git se
ejecutan como `yo` en `/home/yo/projects/kza`. `kza` es la cuenta administrativa
y de ejecución del servidor (UID 1000, grupo `sudo`); no se usa para desarrollar
ni compilar.

El 2026-11-05 se autorizó trasladar el entorno de trabajo a una Mac con cuenta
y checkout locales propios. Esta autorización no equivale a una instalación
Mac verificada ni modifica producción. La construcción y validación operativa
de releases conservan el flujo Linux actual hasta verificar una alternativa
compatible. La continuidad está en el
[traspaso a Mac](../runbooks/2026-11-05-mac-development-handoff.md).

El servidor, los servicios y sus datos continúan bajo `kza`. La publicación se
hace al preparar una release. Mientras se desarrolla bajo `yo`, se conserva la
versión que está funcionando en producción, incluidos los archivos que necesita
para ejecutarse. La retirada del checkout y de las copias de desarrollo se
coordina con la sustitución de esa versión en la próxima release.

La release operativa del 2026-10-28 transfirió a `kza` únicamente el artefacto
compilado y sus recursos de ejecución, construidos y probados bajo `yo`.
Secretos y datos persistentes permanecen fuera del artefacto y de Git. El
proyecto es Python: `scripts/build_runtime.py` construye un paquete de bytecode
CPython 3.13 sin fuentes `.py`. El ensayo offline de instalación, promoción y
rollback está en `scripts/stage_runtime.py` y el
[plan de H0](../plans/2026-10-25-h0-baseline-contracts.md). El rescate no cambió
el servicio; la instalación posterior fue solicitada expresamente por el usuario.

El flujo histórico de edición en producción y puente desde la laptop queda
retirado para trabajo nuevo. Consultar `AGENTS.md` en la raíz para las
instrucciones operativas.

## Release operativa instalada — 2026-10-29

Está activo `release_20261029T185008966364Z_e9669ec`, construido desde
`e9669ec1edb101ca6b190ef8df09314e1ac63562` y publicado el 2026-10-29 a las 15:50
(-03) con `tools/publish_voice_loading.py` (registro
`/home/kza/releases/deployments/voice-loading-e9669ec.json`). Mejora la carga de
muestras de Voces sobre `release_20261029T174830709997Z_63ea8d3`, publicado
ese mismo día a las 14:48, que ordena las muestras de Usuarios / Voces por fecha
descendente y verifica la paginación. Ese slot anterior sigue en `slots/` con los
mismos cinco enlaces; desde R1E (2026-11-18) no es destino de L4 (ver «Volver al
contenedor de e9669ec») y el retiro Q5 del día 14 lo borra (R1E-2). El operador publicó el artefacto compilado solicitado por el usuario;
`yo` conserva acceso de diagnóstico, sin shell operativa en `kza`.
Esta instalación no acepta H6 ni cierra el roadmap. La última suite completa
mantiene dos fallos conocidos. La observación posterior confirma captura en Hall
y Escritorio; Living y Cocina siguen sin captura útil. HA conserva `auth_error`.

Observación del 2026-11-17: la Cocina no capturaba porque su micrófono enumeró
el 2026-11-15, después del arranque de `kza-voice` (2026-11-02), y PortAudio no
relee su lista de dispositivos. Con autorización del usuario se reinició
`kza-voice` como `kza` a las 00:00:19; la Cocina volvió a entregar audio a las
00:00:57 y las cuatro habitaciones quedaron capturando. El refresco automático
del catálogo (T025-U2, `rooms.stream_watchdog.catalog_refresh_min_interval_s`)
está desarrollado, sin release; hasta instalarlo, un micrófono que enumera
después del arranque requiere reiniciar el servicio. Detalle en
[aceptación de la Cocina](../runbooks/2026-11-16-cocina-acceptance.md).

- Runtime: `/home/kza/releases/current` apunta al subdirectorio `runtime` del
  slot instalado. El 2026-11-18, sólo leyendo, se verificaron los 362 archivos de
  su manifiesto (326 módulos compilados, sin fuentes `.py`); respecto de 63ea8d3
  cambian sólo dos assets JS del dashboard y `.pyc` de `tools/`, ningún `src/*.pyc`.
- Estado: `state`, hermano de `runtime`, es un contenedor privado cuyos enlaces
  conservan las rutas canónicas de configuración, datos, entorno, modelos y logs
  del slot `release_20261028T145442Z_b0866e8`. No se movieron ni reemplazaron datos:
  los registros de retención conservan sus localizadores absolutos. El directorio
  de trabajo es el nuevo contenedor; `runtime/config`, `runtime/data` y
  `runtime/.env` enlazan a él. No retirar el estado del slot anterior: sólo
  lo retira `retire-legacy-state` desde DÍA 0 + 14 × 24 h, contado desde el
  primer cierre limpio gestionado (ver «Retiro del estado congelado de b0866e8 (Q5, P7)»).
- `kza-voice.service` conserva su unit original y usa el drop-in `release.conf`
  (el lanzador gestionado U3b-0d lo reemplazará en R1; ver «Lanzador gestionado (U3b-0d)»).
  Ejecuta `/home/kza/kza/.venv/bin/python -B -P -m src.main`, con
  `PYTHONPATH=/home/kza/releases/current`. El venv y los modelos existentes
  permanecen externos; no se instalaron dependencias ni se reasignaron GPUs.
- Actualización: sólo cambió `WorkingDirectory` en el drop-in existente; el
  selector de código continúa siendo `PYTHONPATH=/home/kza/releases/current`.
  Se reinició únicamente voz. El proxy y la rotación de consola conservaron su
  ejecución. Cada publicación del 2026-10-29 dejó un respaldo consistente de las
  mismas diez bases SQLite del estado de b0866e8:
  `releases/backups/release_20261029T174830709997Z_63ea8d3/` (14:48) y
  `releases/backups/release_20261029T185008966364Z_e9669ec/` (15:50).
- Prueba de Hall: el parámetro RAM `PP_MIN_NN=0.51` fue repuesto y verificado tras
  el arranque, con los demás parámetros estáticos iguales. La configuración
  persistente sigue en `0.4`; la mejora subjetiva continúa pendiente de comparar.
- Dashboard: **https://192.168.1.2:9540/obs/**. NGINX del proyecto, bajo
  `kza-dashboard-proxy.service`, termina TLS y reenvía a `127.0.0.1:9500`.
  Conserva Host, esquema y WebSocket. No modifica el NGINX compartido.
- Pendiente de release (no instalado): en la rama de desarrollo el dashboard es
  NEXO en `/` y el panel técnico `/obs/` queda sólo por URL directa, sin enlace
  desde NEXO (decisión D1 del usuario, 2026-11-10). La release instalada no
  contiene NEXO y sigue usando `/obs/`.
- Pendiente de release (no instalado): T040/T041 U4 retira en la rama de
  desarrollo `/home/legacy.html`, `/home/home.css`, los `routines.js` y
  `memory.js` heredados y el dashboard raíz `/index.html` (`dashboard.js`,
  `style.css`); Rutinas y Memoria quedan en NEXO. **La release instalada
  todavía sirve esos archivos** y los conserva hasta que una release los
  sustituya. Ojo: la release instalada (slot activo e9669ec) no tiene
  `legacy.html`; allí las vistas heredadas son `/home/` mismo (`home/index.html`
  con `home.js`, `home.css`, `routines.js`, `memory.js`) y la grilla raíz
  (`/index.html`, `dashboard.js`, `style.css`). En desarrollo, con
  `dashboard.home.enabled: false`, `/` responde 404 con una página que explica esa
  opción, sin enlaces; los archivos estáticos de `/home/` siguen montados.
- El certificado es local y autofirmado. El login usa el
  `KZA_DASHBOARD_TOKEN` existente de `/home/kza/secrets/.env`; no copiarlo a Git
  ni al chat. El acceso privado sin sesión devuelve 401.
- Voz, proxy y timer de rotación están habilitados; `kza` tiene linger
  configurado. Se verificó su estado actual, sin ensayo de reinicio del host.
- Durante la instalación, el journal global bloqueó la escritura de stdout de
  KZA. El drop-in dirige ahora la consola al archivo privado
  `state/logs/console.log`. `kza-console-rotate.timer` ejecuta logrotate cada
  hora, con 24 rotaciones, `maxage 1`, compresión y `copytruncate`. El usuario
  reinició journald y su CPU volvió a subir. El diagnóstico posterior identificó
  una avalancha de captura USB; retirar sólo el micrófono involucrado recuperó
  journald sin otro reinicio, con una ventana estable de más de tres minutos.
  El defecto de la ruta USB sigue pendiente. La salida privada evita depender
  del socket del journal y se conserva en esta release.

La configuración y los datos originales de `/home/kza/kza` conservan sus hashes.
El respaldo privado está en
`/home/kza/releases/backups/release_20261028T144002Z_2c5adb4/`; ese identificador
nombra la operación de respaldo, no la versión original de producción.
El checkout original y `/home/kza/app` se conservan para rollback y para el
índice de código existente. La limpieza de fuentes sigue pendiente.

Ver [operación y rollback](../runbooks/2026-10-28-compiled-runtime.md) y la
[evidencia de actualización del 29](../research/2026-10-29-voice-order-publication.json).
La [instalación original del 28](../research/2026-10-28-live-release-validation.json)
se conserva como antecedente y como versión de código para rollback.
El acceso posterior de consulta desde `yo` usa una clave con comando forzado,
sin shell ni permisos de operación; ver
[diagnóstico restringido](../runbooks/2026-10-28-readonly-diagnostics.md).
La investigación posterior atribuyó el atasco de journald a una avalancha de
avisos de captura USB; el operador aisló únicamente el micrófono identificado.
Ver [prueba de aislamiento y límites](../runbooks/2026-10-28-journald-usb-flood.md).

## Construcción del artefacto

La mejora de carga de muestras de Voces desde
`e9669ec1edb101ca6b190ef8df09314e1ac63562` está instalada desde el 2026-10-29
15:50 (ver «Release operativa instalada»). Cambia únicamente dos assets de la
aplicación: hace visible el reproductor al seleccionar una muestra, muestra el
progreso y conserva el audio seleccionado al repetir la selección. El diagnóstico
por HTTPS desde el servidor midió 5–11 ms para ocho muestras; no reproduce la red
ni Chrome en Windows del usuario. Ver la
[evidencia y límites](../research/2026-10-29-voice-loading.json) y el
[comando de publicación](../runbooks/2026-10-28-compiled-runtime.md#mejora-de-carga-de-muestras--instalada-2026-10-29).

La release liga el dashboard a `127.0.0.1:9500` detrás del proxy del proyecto.
Todo endpoint HTTP/WS privado requiere `KZA_DASHBOARD_TOKEN`,
incluidos health y observabilidad. La variable de ejemplo está vacía; el valor
real permanece en el entorno privado de ejecución. La shell de login y los
assets son públicos, sin datos del hogar. No se expone OpenAPI por una ruta
automática. Los clientes de diagnóstico deben enviar Bearer o sesión válida.

El proxy debe preservar Host y establecer correctamente el
esquema externo para que se validen Origin y la cookie Secure en HTTPS. Sólo se
confía en forwarding desde el proxy local; no se toman headers de cualquier
cliente como prueba de identidad. Las pruebas ASGI comprueban la frontera, pero
la instalación del 28 de octubre verificó el login, la cookie Secure y el canal
en vivo en navegador. Esa comprobación no sustituye la aceptación de H6.

Desde el checkout de `yo`, con CPython 3.13:

```bash
python3.13 scripts/build_runtime.py dist/kza-runtime-cp313.tar.gz
```

La construcción toma únicamente la revisión commiteada (`HEAD` por defecto,
seleccionable con `--revision`) y conserva el SHA en `release.json`. Compila los
módulos a `.pyc` importables sin fuentes y agrega los recursos del frontend.
Excluye configuración, `.env`, dependencias, modelos, datos, tests y Git. No
reemplaza un archivo de salida existente. La construcción debe producir el mismo
archivo para el mismo commit y versión de CPython. El ensayo T024-E detectó
diferencias entre procesos de la misma revisión; esa
[evidencia histórica](../research/2026-11-05-compiled-application-restore.json)
se conserva. T024-F0 reprodujo un contracaso por el estado de internado del
llamador y separó ese efecto del parseo y la retención de cadenas. El builder
ahora compila en un proceso nuevo con `PYTHONHASHSEED=0`, `py_compile` estándar
y comprobación de ABI e inventario. El archivo completo coincide byte por byte
en la matriz y tandas verificadas con CPython 3.13.18 arm64; conserva errores,
warnings y la publicación sin sobrescritura. El requisito de reproducibilidad
sigue vigente para cada plataforma soportada. Ver el [alcance probado y sus
límites](../research/2026-11-06-compiler-process-reproducibility.json).

El baseline local T024-F1 confirmó que `prepare_slot`/`activate_slot` aún permiten
seleccionar un estado viejo sin cuatro decisiones de autoridad actuales. La
integridad del artefacto y el emparejamiento de rutas no prueban un rollback
seguro de datos y permisos. La regresión nueva sigue fallando, sin corrección de
producto; ver [evidencia y continuación](../research/2026-11-06-copied-slot-authority-baseline.json).

En desarrollo, sin release (U3b-0a): una activación gestionada valida antes de
mover el head el slot candidato con su propio `scripts/stage_runtime`, en un
proceso de sólo lectura. Un artefacto sin el par `scripts/stage_runtime.pyc` +
`src/core/runtime_admission.pyc` (por ejemplo 9b31702, 63ea8d3 o b0866e8) se
rechaza como `runtime_admission_candidate_unmanaged` sin cambiar nada. Por eso
los ensayos compilados de slots copiados ya no fijan 9b31702: usan el ancestro
más antiguo de HEAD con ambos compañeros, derivado con git
(`tests/mocks/slot_rollback.managed_revision`; hoy 80b47f5).
`KZA_SLOT_ROLLBACK_REVISION` sólo acepta revisiones con ese par.

Consecuencia operativa: en cuanto exista un head gestionado, volver al slot de
producción b0866e8 (o a 63ea8d3 o e9669ec) queda rechazado; sólo una activación no
gestionada, sin head ni autoridad, los acepta todavía. El rollback posterior
debe apuntar a una release que incluya el par.

Lista de releases (G01, 2026-11-14): no promover ni volver a una release
gestionada anterior a 431e44e. Esas versiones (por ejemplo 80b47f5) tienen el
par, pero su mantenimiento de diagnósticos puede tomar el lock del catálogo en
el estado delegado durante la revalidación final y rechazar el arranque con
`runtime_admission_writer_active` (`t024/g01-writer-active-20261114.json`).
Ninguna se publicó; el ensayo compilado que usa 80b47f5 lo compensa sólo en el
harness (`tests/mocks/managed_slot_graph.serialize_older_history`).

La sonda corre con el intérprete del controlador, `-I -B`, sin el lease, sin
ticket y con un entorno mínimo (`LANG`, `LC_ALL`, `LC_CTYPE`, `TMPDIR`,
`SQLITE_TMPDIR`). Que sea de sólo lectura es una convención del código del
candidato, no un sandbox. Su límite es `activate --probe-timeout` (por defecto
60 s; sólo 1–3600 s, otro valor se rechaza como
`runtime_admission_probe_timeout`). Cada activación gestionada y cada `run()`
pagan un segundo snapshot completo. En un Mac M-series midió 2,6 s con 0,49 GB
de bases sintéticas y 10,4 s con 1,95 GB, unos 5,3 s/GB. Antes de la release hay
que medirlo con una copia de tamaño real bajo `yo` en el servidor
(`tests/mocks/probe_cost_harness.py`).

### Delegación retirada desde SC-2 (en desarrollo, sin release)

Decisión del usuario del 2026-11-15 (`u3b_f8_self_contained_releases`,
diseño `sc-release-design-v3`): las releases gestionadas pasan a ser
autocontenidas y la delegación a estados dueños se apaga. Desde SC-2
(época del controlador 4) el controlador no concede nada; las copias viejas de
`stage_runtime` son otra cosa (ver más abajo):

- `stage_runtime --root R --offline delegate <slot>` imprime
  `{"refused": "runtime_admission_delegation_retired"}` y sale con 78, sin
  tomar el lease ni escribir nada. La entrada del parser se conserva para que
  el operador vea el código y no un error de argparse.
- `delegate_slot`, `authorize_target_delegation` y `authorize_derived_state`
  (también los métodos de `AdmissionController`) rechazan con
  `runtime_admission_delegation_retired` antes de cualquier lectura, lock o
  escritura. No se escriben `admission/delegations/`,
  `admission/delegation-chains/`, `admission/derived-authority/` ni la
  política del destino.
- `activate` de otro slot bajo un head gestionado rechaza con
  `runtime_admission_transition_requires_release_mode` antes de leer el
  candidato: no guarda binding, marker, CAS ni selección. Sólo se activa el
  slot del propio head (lo hacen `serve` y `run`) o, bajo la compuerta de su
  compactación, el K de ese log. Una raíz sin head (no gestionada) no cambia.
  Sin delegación, un `prepare` + `activate` de una copia sin rutas podía
  convertirse en head y dejar atrás a H; esta regla hace que, desde R1, la
  única transición gestionada entre slots sea el modo release.
- Siguen moviendo el head, por diseño: `adoption-commit` cuando el head es un
  head de adopción abortado (re-adopción desde el legado), el K de una
  compactación simple (el mismo archivo que H por P6; A1 rechaza un K
  delegado) y el regreso a H de `compact abort --to-h`
  (`compaction_return_to_h`, bajo la compuerta de su compactación). Ninguno
  concede dueños ni autoridad derivada.
- Una copia de `stage_runtime` que ya contiene `require_forward_epoch`
  (desde b0c322d, la corrección F8 de U3b-0c, todavía época 3; el 26de262
  fijado incluido), sea de un slot o de `controller/<commit>`, no puede tomar
  el lease ni preparar en una raíz cuyo `admission/controller.json` declare
  época 4 (`runtime_admission_controller_downgrade`). `install-controller`
  rechaza cualquier archivo de época menor que 4
  (`runtime_admission_controller_epoch_too_old`).
- El bloqueo vive en el código viejo, así que sólo cubre esas copias. Toda
  copia de `stage_runtime` de 69a79e6 (U3b-0a, donde entra `delegate`) a
  b0c322d~1 no comprueba la época: toda la época 2 (a76fa97..f4c72ed~1) y la
  época 3 de f4c72ed hasta antes de b0c322d. En una raíz de época 4 esa copia
  todavía concede (`delegate` escribe `admission/delegations/` y mueve el
  head) y activa otro slot, que pasa a ser el head que corre `serve` (revisión
  AUTHZ3-1, con las copias de b0c322d~1 y a76fa97). Las releases legadas de
  producción (b0866e8, 63ea8d3, e9669ec) no tienen `delegate`, pero tampoco comprueban
  la época. Por eso no se corre ningún `stage_runtime` que no sea
  `controller/current`, y SC-6a (PS-13) lista y borra o renombra todo
  archivo, árbol extraído y checkout construido desde ese rango, bajo `yo` y
  en cada ubicación de transferencia, no sólo los R1 de época 3. El historial
  de Git conserva esas versiones: para ellas la única barrera es esa regla de
  operación.
- Barrera estructural diferida a SC-1 (decisión técnica del 2026-11-15): una barrera de época 4
  que no dependa del código viejo. `install-controller` o `adoption-commit`
  congela el conjunto de capacidades de delegación existentes, y la
  validación, `serve` y la compactación rechazan cualquier registro de
  `admission/delegations/` fuera de ese conjunto. No cubre un `prepare` +
  `activate` sin concesión desde una copia vieja; para eso `serve` tendría que
  aceptar sólo heads cuyo binding registró un controlador de época 4. Exige
  una enmienda del diseño, con su test y su mutante; entra con SC-1, que
  rediseña los bindings y sus heads aceptados. Hasta entonces R1 queda
  cubierta por la regla de operación (sólo `controller/current`) y la
  limpieza de SC-6a: las releases legadas no tienen `delegate` y, bajo un
  head gestionado, `run()` sólo corre el slot del head, así que mover
  `current` no cambia lo que corre.
  SC-1a (2026-11-16, desarrollado, sin release) la implementó de otra forma,
  según el diseño v4 aceptado (`mechanism.authz3_1`): una guarda sin estado
  de época 5, sin conjunto congelado ni registro de heads. Ver «Controlador
  de época 5 (SC-1a)» más abajo; R1 (época 4) no la tiene.
- Ese bloqueo necesita un `admission/controller.json` legible: si falta, es un
  symlink, no se puede leer o no es JSON válido, `require_forward_epoch` no
  rechaza nada e incluso una copia que lo contiene podría volver a conceder y
  activar otro slot (residuo aceptado en el diseño,
  `delegation_off.old_controller_copies`, que no menciona las copias sin la
  comprobación).
  Con el registro dañado, repararlo sólo con `controller/current` o
  `install-controller`, y no correr el `stage_runtime` de ningún slot hasta
  que vuelva a declarar la época 4.

#### SC-6a: ensayo ampliado y copias viejas retiradas (2026-11-15)

- R1: construido bajo `yo` en el server desde 81ed568 (`controller_epoch` 4,
  `schema_contract` 1), dos veces, idéntico byte a byte; la construcción en la
  Mac (CPython 3.13.18 contra 3.13.12 del server) da los mismos bytes. sha256
  `3f57bd197f2f1633f32683616abefd3db038466d7a966d9925b24c84fc342a0e`; ese
  archivo quedó reemplazado (abajo) y renombrado
  `/home/yo/projects/releases-r1/r1-81ed568.tar.gz.superseded-r1b`.
- R1 re-fijado tras R1-B1/R1-B2 (2026-11-15): las dos cambian `src/` y el
  frontend, así que R1 se reconstruyó igual desde c6462e5 (`controller_epoch`
  4, `schema_contract` 1, 416 módulos compilados, 476 archivos): dos veces en
  el server, idéntico byte a byte, y una en la Mac, igual. sha256
  `733b6fc35dc929accc00b78b231092531a7ca131663ccc8b663f9d9e48851b86`, en
  `/home/yo/projects/releases-r1/r1-c6462e5.tar.gz` (0600, directorio 0700,
  con `SHA256SUMS`). Frente al de 81ed568 cambian sólo `release.json` y los
  siete archivos de R1-B en `src/`. Ese archivo quedó reemplazado (abajo) y
  renombrado `r1-c6462e5.tar.gz.superseded-rr2-1`.
- R1 re-fijado tras la re-revisión (2026-11-15): el arreglo RR2-1 cambia
  `src/memory/capture_recorders.py`, así que R1 se reconstruyó igual desde
  357dcb5 (`controller_epoch` 4, `schema_contract` 1, 416 módulos
  compilados, 476 archivos): dos veces en el server, idéntico byte a byte, y
  una en la Mac, igual. sha256
  `c5305d588a228aca445489b90c00a2b11165d24f8fa1078ad19a68fbe9630a24`, en
  `/home/yo/projects/releases-r1/r1-357dcb5.tar.gz` (0600, directorio 0700,
  con `SHA256SUMS`). Frente al de c6462e5 cambian sólo `release.json` y
  `src/memory/capture_recorders.pyc`. El runbook de R1 fijó ese sha256 y su
  commit hasta R1E (abajo); ese archivo sigue al lado con su nombre.
- R1 re-fijado en R1E (2026-11-18): adoptar el contenedor de e9669ec y la
  ronda 1 de la revisión independiente de R1E (R1E-RV3: commit completo del
  contenedor; R1E-RV4: ruta de logrotate entre comillas) cambian
  `scripts/`, así que R1 se reconstruyó igual desde d045156
  (`d045156bd8922c5c6f2b6364827a314a5bc50da0`, `controller_epoch` 4,
  `schema_contract` 1, 416 módulos compilados, 476 archivos): dos veces en
  el server, idéntico byte a byte, y una en la Mac, igual. sha256
  `b2205650f045ece0058c05db02b03c5bdb6523e4ad8b229fbbcd702685133990`, en
  `/home/yo/projects/releases-r1/r1-d045156.tar.gz` (0600, directorio 0700,
  con `SHA256SUMS-d045156`). Frente al de 357dcb5 cambian sólo
  `release.json`, `scripts/adoption.pyc`, `scripts/adoption_commands.pyc` y
  `scripts/managed_launcher.pyc`. Una primera reconstrucción desde 695708a
  (`r1-695708a.tar.gz`, sha256 `d6e5a21a…`) no se fijó. Los archivos de
  357dcb5 y 695708a quedan al lado sin renombrar y no se transfieren. El
  runbook de R1 fijó el sha256 de d045156 y su commit hasta R1E-2 (abajo);
  antes de W1 exige que `sha256sum "$R1_ARCHIVE"` coincida con el fijado o se
  detiene. Que ese commit adopte
  el contenedor de `C` lo comprueba un test
  (`test_r1e_the_pinned_r1_admits_the_runbook_container`): un R1 anterior a
  R1E deja pasar el contenedor de e9669ec en la corrida de prueba, W1b, W2
  y el commit de W4, W-abort-1 salta su comprobación, y sólo W-abort-2, L4
  y Q5 lo rechazan.
- R1 re-fijado en R1E-2 (2026-11-18, decisiones `r1e_questions_20261118`):
  el retiro Q5 del día 14 con el alcance de producción (informes, veinte
  copias, slots de 63ea8d3 y 2c5adb4) y L4 sólo hacia el contenedor de
  e9669ec cambian `scripts/`, así que R1 se reconstruyó igual desde 7752570
  (`7752570fb0e4812c5888770111eb2da59d12877a`, `controller_epoch` 4,
  `schema_contract` 1, 416 módulos compilados, 476 archivos): dos veces en
  el server, idéntico byte a byte, y una en la Mac, igual. sha256
  `bec0b82dbfaf4c13af543e5cab7bb81315550d8f2136cd9c948dae34131dc3ce`, en
  `/home/yo/projects/releases-r1/r1-7752570.tar.gz` (0600, directorio 0700,
  con `SHA256SUMS-7752570`). Frente al de d045156 cambian sólo
  `release.json`, `scripts/adoption_commands.pyc` y
  `scripts/managed_launcher.pyc`. `r1-d045156.tar.gz` queda al lado sin
  renombrar y no se transfiere. El runbook de R1 fija el sha256 de 7752570 y su
  commit; `test_r1e2_the_pinned_r1_retires_production_s_day14_layout`
  comprueba que ese commit trae el alcance del día 14 de R1E-2.
- Ensayo: `tests/integration/test_sc6a_r1_rehearsal.py` (en proceso, Mac y
  server) y la ventana real de `tests/integration/test_linux_adoption_window.py`
  agregan los casos (i)–(iv) del diseño. Después de W9, `delegate`,
  `authorize_derived_state` y un `activate` de otro slot rechazan. El R1 de
  época 3 (26de262) lo rechazan el prepare de R1 y `install-controller`, antes
  y después de W3. `read_r1_archive` rechaza un archivo de época 5 y uno de
  época 3. El bloque del sha256 fijado del runbook de R1, corrido tal como está
  escrito, detiene el procedimiento con el R1 de época 3 y con una
  reconstrucción distinta del mismo commit.
- AUTHZ3-1 ensayado: una copia construida desde b0c322d~1, sobre la raíz
  adoptada de época 4 después de W9, prepara un slot, concede
  (`admission/delegations/`) y lo activa: ese slot pasa a ser el head. La
  primera copia con `require_forward_epoch` (b0c322d) se rechaza con
  `runtime_admission_controller_downgrade`. El test fija ese residuo tal como
  es. La barrera sigue siendo la regla de operación (sólo
  `controller/current`), la limpieza de abajo y, más adelante, la guarda de
  SC-1.
- Inventario y retiro bajo `yo` (Mac y server; cada ubicación está en el
  handoff y en `t040-t041/sc6a-evidence.json`): se borraron los árboles
  temporales de pytest, las corridas del roadmap y los restos de
  `build_runtime` con copias de 69a79e6..b0c322d~1 o de época 3. El checkout
  del server avanzó de 26de262 a la rama integrada. El primer inventario
  buscaba sólo por nombre y no vio cuatro copias sueltas renombradas en la Mac
  (`/private/tmp/sr_head.py` y su gemela en el tmp del roadmap, del blob de
  389a18c..4bb38e3~1; `stage_runtime.r.backup.py`, de 69a79e6; y
  `stage_runtime.backup.py`, una copia de trabajo sin commit del 2026-11-10),
  todas con `delegate` y sin `require_forward_epoch`. La revisión (SC6A-R1) las
  encontró; eran restos del roadmap y se borraron. El inventario se repitió
  por contenido: cada archivo con `def delegate_slot(` o cada `.pyc` que lo
  nombra, y cada miembro de cada archivo comprimido, clasificados por
  `require_forward_epoch` y `delegation_retired`, en la Mac, en el server bajo
  `yo` y, sólo leyendo, del lado de `kza`. No queda ninguna copia de ese rango
  ni de época 3. No apareció nada de origen dudoso, así que no hubo renombres
  `*.superseded-epoch3`. Del lado de `kza` (sólo listado, sin cambios):
  ningún archivo, árbol ni checkout de ese rango. Sus slots e `incoming/` son
  las releases legadas 2c5adb4, b0866e8, 63ea8d3 y e9669ec, sin `delegate`.
- Lo que queda: el historial de Git conserva esas versiones, y la propia suite
  construye copias de ese rango en el tmp privado de pytest de `yo` (238b463
  en el multi-hop compilado, 40aa05b en el ensayo del lanzador, b0c322d~1 en
  este ensayo; pytest guarda las tres últimas corridas). No son ubicaciones de
  transferencia: R1 se toma sólo del archivo cuyo sha256 fija el runbook de R1
  (paso obligatorio antes de W1).

#### R5V-1: dueños de captura arrastrados de un arranque anterior (2026-11-15)

Hallazgo R5V-1 de la revisión del diseño v4 (`t040-t041/sc-release-design-v4.json`).
El token de un grabador (navegador o cliente nativo) vive sólo en la memoria
de la página o del proceso. `_initialize` pasa cada fila `active` a `unknown`
en cada arranque. Si la pestaña se cerró sin `retire` durante un reinicio,
nadie puede volver a retirar esa fila. Hasta 81ed568, `drain()` esperaba su ACK:
toda parada gestionada quedaba colgada hasta el SIGKILL de `TimeoutStopSec`,
sin recibo de cierre (head `UNKNOWN` en cada parada, y la compactación y R1
necesitaban recuperación cada vez). Además, un mute total quedaba en
`muting` y el unmute se rechazaba con `capture_control_busy`.

Arreglo técnico (opción e, rama `r1b1-orphan-recorder`, sin release). Es una
**decisión técnica**, no un reconocimiento pendiente del usuario. La regla del
2026-11-04 (`docs/research/2026-11-04-global-capture-control.json`) es un
registro técnico: exige un recibo de parada explícito de todo dueño
`active`/`unknown`. El arreglo conserva su parte de confirmación («Unknown/foreign
participants cannot receive confirmation»: `total_mute_confirmed` sigue falso
mientras exista uno). Sólo deja de exigir ese recibo a la parada, al cierre y a
la fase `muted` cuando el dueño quedó `unknown` al arrancar y no volvió a
hablar con el proceso. Con eso restituye la garantía H0 del usuario: el mute
siempre se puede reactivar desde el control autenticado. La versión
anterior de este texto (5b40447) lo presentaba como un visto bueno
bloqueante del usuario; no lo es.

- Las filas que ya estaban `unknown` al arrancar este proceso quedan como
  *arrastradas* (`carried`). `drain()`, el `pre_close` del testigo y
  `verify_closed` esperan sólo `pending_live()`, que no las incluye. El
  servidor no les concede activación ni permiso de captura (`poll` da
  `capture_allowed: false`). Las subidas no dependen del token del grabador:
  `capture_ingress` y `CaptureControl.admit_upload` las admiten por la sesión
  de administrador y la política global, así que una pestaña que sobrevivió
  sólo deja de subir por su propio sellado (`sealed`) del lado del cliente.
- Premisa: el servidor no distingue un token perdido de un titular que
  sobrevivió al reinicio (una pestaña abierta o un cliente nativo como
  `scripts/train_wakeword.py` o zone_setup). No esperar a ese titular depende
  de los monitores de cierre seguro de cada cliente: el nativo consulta cada
  0,25 s con 1 s de timeout y el navegador cada 0,5 s con 5 s de timeout; ante
  un fallo o `retirement_required` cada uno sella su captura.
- Un titular que presenta su token a `poll` o `activate` (con su tipo, antes
  de `aclose`) demuestra que está vivo: deja de estar arrastrado y las drains
  vuelven a esperar su `retire`, que el cliente envía al ver
  `retirement_required`. Sólo quedan sin esperar los titulares que nunca
  contactan al proceso nuevo. Si uno contacta después de que la drain del
  cierre terminó y no retira, el cierre se rechaza (sin recibo), como con
  cualquier dueño vivo.
- El silencio nunca cuenta como ACK. `pending()` sigue contándolas: mientras
  exista una, `total_mute_confirmed` es falso (el panel no muestra «Captura
  global pausada y confirmada»). La prueba de cierre del testigo
  (`verify_closed`) da `capture_permissions_retired: false` con
  `capture_permissions_carried: N`, pero sólo en memoria: el recibo de
  admisión guarda únicamente `graph_closed`, `privacy_closed`, `sdk_closed` y
  `native_closed`, así que se escribe igual y no se distingue de uno limpio.
  Lo único observable es una advertencia sin contenido que el cierre registra
  antes de cerrar el logging (`[Admission] closing with N carried capture
  permission(s) never acknowledged; the receipt does not record them`) y las
  filas de `capture_browser_recorders` en el estado. Guardar el conteo en el
  recibo es un cambio de esquema que queda para SC: el diseño v4 lo prevé en
  modo verificación con la misma clave, `capture_permissions_carried`.
- Los dueños emitidos por el proceso actual mantienen la barrera completa de ACK
  explícito. La capacidad de 256 cuenta sólo esos dueños vivos. Al arrancar se
  registra únicamente el conteo de filas arrastradas, sin tokens ni hashes.
- Residuo: un dueño vivo (emitido por el proceso actual o reclamado) cuya
  pestaña o cliente muere sin `retire` sigue bloqueando la parada de ese
  proceso hasta el SIGKILL de `TimeoutStopSec`: sin recibo, head `UNKNOWN` y
  recuperación para la compactación o R1. Es el síntoma de R5V-1, ahora una
  vez por dueño perdido y no en cada parada. En el arranque siguiente esa
  fila pasa a arrastrada. Mitigaciones:
  - el estado de captura (`GET /api/capture/control`) informa
    `live_capture_owners` (los que una parada esperará) y
    `carried_capture_owners` (los que no); el panel de Privacidad muestra
    ambos;
  - los runbooks de R1 y de compactación exigen `live_capture_owners` = 0
    justo antes de la parada gestionada;
  - la página reintenta cada 2 s un `retire` fallido mientras siga abierta
    (antes sólo lo reintentaba en `pagehide`).
- Filas arrastradas sin tope: la capacidad sólo cuenta dueños vivos, así que
  cada caída con N dueños activos suma N filas permanentes. `_read_cache` y
  `_register` las recorren en cada mutación (costo lineal), y nada las acota
  ni alerta salvo la advertencia de arranque. Con una sola, la pausa total no
  se confirma hasta que un administrador las olvida (R1-B2, abajo). Antes
  quedaba en `muting`; ahora llega a `muted` sin «confirmada». Un tope o un
  umbral de alerta siguen sin existir.

#### R1-B2: «Olvidar grabaciones perdidas» (2026-11-15)

Decisión del usuario del 2026-11-15 (`r1b1_lost_recordings` en
`t040-t041/user-decisions-20261110.json`, aceptada): las grabaciones perdidas
(dueños arrastrados) las resuelve un administrador desde Privacidad con
«Olvidar grabaciones perdidas», afirmando que no queda ninguna pestaña
grabando. Queda un registro sin contenido. Hasta entonces el panel muestra
cuántas hay y nunca dice «confirmada». Rechazadas: el retiro automático al
arrancar y conservarlas para siempre.

- Ruta: `POST /api/capture/control/forget-carried` (rama
  `r1b1-orphan-recorder`, sin release). Cuerpo estricto
  `{"expected_carried": N, "no_capture_running": true}`: `N` es el conteo
  `carried_capture_owners` que la página mostró y la afirmación tiene que ser
  literalmente `true`. Campos extra, claves duplicadas, tipos laxos (`1`,
  `"true"`, `2.0`, `true` como conteo) o un cuerpo que no es JSON dan 422/415
  sin tocar nada.
- Permiso: `admin:manage`, el mismo del control de captura global, en la capa
  de sesión (`_route_permissions` para `/api/capture/*`) y otra vez al crear el
  `CaptureController`. Ningún rol de hogar ni de evaluación lo tiene; la voz o
  la persona reconocida no eligen permisos. La autoridad del administrador se
  vuelve a comprobar dentro de la transacción, también en la última
  comprobación previa al commit: si vence antes de esa comprobación, se
  revierte (403). Lo prueban tests de integración a través de
  `CaptureControl.forget_carried` y de HTTP con una sesión que vence recién
  cuando la transacción ya eligió las filas (R1B2-1).
- Sólo olvida dueños **arrastrados**: filas `unknown` al arrancar este proceso
  que ningún titular reclamó presentando su token. Nunca toca dueños vivos
  (emitidos por este proceso o reclamados), ni reservas. Si el conteo actual
  difiere de `N`, responde 409 `capture_carried_count_conflict` y no olvida
  nada, así que una página vieja no puede olvidar dueños que no mostró. Si un
  titular presenta su token mientras la transacción hace sus comprobaciones,
  el olvido se revierte entero (409) y ese dueño pasa a vivo. El reclamo no
  toma el lock del ledger, así que puede caer entre la última comprobación y
  el commit (R1B2-3): en ese caso el olvido se confirma para los demás y la
  fila de ese titular se restituye como `unknown` en una transacción
  inmediata, antes de que ninguna lectura de caché ni `retire` vea el borrado.
  Queda como dueño vivo: las drains y `total_mute_confirmed` esperan su
  `retire`, la respuesta no lo cuenta en `forgotten_capture_owners` y el log
  agrega `[CaptureRecorders] N carried owner(s) contacted this process as a
  forget committed; kept as live owners`. Un titular que se presenta después
  de esa restitución ya fue olvidado y recibe `retirement_required`.
- Un olvido que se cruza con el cierre se rechaza (503, re-revisión RR2-1,
  2026-11-15). Desde que empieza el `aclose` de `CaptureRecorders`, un token
  presentado ya no se reclama, así que el olvido no podría ver a un titular
  que volvió. Si el cierre empezó antes de una comprobación de la
  transacción, el olvido se revierte; si empezó entre la última comprobación
  y el commit, todas sus filas se restituyen como `unknown` y el log agrega
  `[CaptureRecorders] shutdown began as a forget committed; its N carried
  owner(s) were restored`. En los dos casos no se olvida nada, la prueba de
  cierre sigue informando esos dueños arrastrados y el rechazo no hace fallar
  el cierre (antes, un olvido rechazado durante el `aclose` lo hacía fallar).
- Dos olvidos simultáneos (dos pestañas de administrador, una vieja) se
  serializan: el lock del ledger cubre todo el olvido, su limpieza y la
  restitución. Así, uno que falla no puede borrar el conjunto que el otro
  está olvidando, y un titular que vuelve durante el segundo lo revierte entero
  (409) en vez de quedar restituido (R1B2-2). Con la restitución de R1B2-3, el
  lock es una segunda defensa: sin él, el olvido se confirma y la fila del
  titular se restituye como dueño vivo (re-revisión RR2-2).
- Idempotente: sin dueños arrastrados devuelve 200 con
  `forgotten_capture_owners: 0`, cualquiera sea `N`, y no registra nada.
- Respuesta: el estado de captura más `forgotten_capture_owners`. Si no queda
  otro dueño ni trabajo, `total_mute_confirmed` pasa a verdadero por las
  reglas normales (fase `muted`, persistida, sellada, soportada, sin
  pendientes). La prueba de cierre en memoria da entonces
  `capture_permissions_retired: true` y `capture_permissions_carried: 0`, y el
  cierre ya no registra la advertencia de dueños arrastrados.
- Registro sin contenido: **sólo log**. `CaptureRecorders` registra
  `[CaptureRecorders] an administrator forgot N carried owner(s), affirming
  none is still recording` (nivel WARNING): el conteo, sin tokens, hashes ni
  el sujeto del administrador. No se usa ninguna tabla: el contrato de
  esquema de la época 1 está congelado (sin tablas, columnas ni valores de
  enumeración nuevos), el ledger de borrados tiene un vocabulario cerrado y
  la auditoría SQLite está apagada en R1. El registro dura lo que dure el
  journal del servicio.
- Las filas se borran de `capture_browser_recorders` con `state='unknown'`;
  no hay cambio de esquema. Un titular olvidado que vuelve a consultar recibe
  `state: retired` y `retirement_required: true`, y su `retire` es un no-op.
- Panel de Privacidad (Captura global): con dueños arrastrados muestra
  «Grabaciones perdidas: N» y, en fase `muted`, «Entrada sellada; la pausa no
  se confirma mientras haya grabaciones perdidas». El botón «Olvidar
  grabaciones perdidas» abre una confirmación con la casilla «Confirmo que
  ninguna pestaña ni herramienta sigue grabando» y el botón «Olvidar N
  grabaciones perdidas», que envía el conteo mostrado. Si el conteo cambia
  mientras la confirmación está abierta, se cierra; un clic sobre la
  confirmación vieja todavía en pantalla envía el conteo que mostraba, y el
  servidor responde 409 si ya no coincide. Un cambio (olvidar, pausar o
  reactivar) pulsado mientras corre la consulta periódica de 1 s se envía en
  cuanto termina; un segundo cambio mientras otro está pendiente se rechaza
  con «Ya hay un cambio en curso. Esperá su resultado antes de elegir otro.»
  (R1B2-7). El panel rechaza un estado que diga confirmada con dueños vivos o
  arrastrados.
- Defensas CSRF de la ruta: cookie `SameSite=strict`, chequeo de `Origin` en
  `require_dashboard_session` y exigencia de `Content-Type: application/json`.
  Un test con sesión de operador (que sí tiene `admin:manage`) prueba que un
  `Origin` ajeno da 403 y un cuerpo de formulario o `text/plain` da 415 sin
  tocar nada (R1B2-4).
- Límite: el olvido no detiene ninguna captura. Se apoya en la afirmación del
  administrador y en los monitores de cierre seguro de los clientes (el
  servidor ya no concedía nada a esos dueños).

#### Contrato de esquema (época 1)

SC-2 declara `SCHEMA_CONTRACT_EPOCH = 1` y `release.json` lo copia como
`schema_contract`. `schema_contract_document()` serializa todo lo que el
controlador exige a un estado: columnas por tabla (compartidas y por base),
marcadores de versión, el conjunto protegido, las reglas de base que aplica el
lector (tablas ignoradas, `memories` sola sin binding, la excepción del SDK,
vistas y triggers rechazados, `application_id`/`user_version` de métricas y
del resto, versiones de `auth_meta`), todas leídas de las mismas constantes
que usa el lector, y los vocabularios de valores. Un test fija su sha256 al
lado de la época: cambiar el contrato sin subir la época falla la suite.

Vocabularios: los cinco de SCv2-P5 (tipos del registro de borrados,
participantes del borrado, estados de hechos, cambios y métodos de dueño) y,
desde la revisión SC2-R1, cada enum de `src` cuyo valor guarda tal cual una
tabla protegida: estados de consentimiento de voz, resultados de
verificación, tipos de evento y motivos de suspensión de atribución, roles
del dashboard, acción, resultado y motivo de `credential_attempts`, estados de
grabadores del navegador, estado, alcance y estado de acción de
recordatorios, estados de corridas y acciones de rutinas, y tipo de dueño de
listas. Desde la revisión AUTHZ-1 la regla es la relevancia para la
privacidad, no el conjunto protegido: además de las tablas protegidas, las
tablas de `privacy.db` que gobiernan el avance del borrado y la atribución
(`erasure_targets`, `erasure_steps`, `policy_index_deletions`, `facts`,
`fact_sources`, no protegidas) fijan cada enum con nombre que guardan:
participantes y estados de hechos (ya en SCv2-P5), `ErasureState` y
`ErasureError` (`erasure_steps.state` y `last_error`), `IndexCleanupError`
(`policy_index_deletions.last_error` y la recomprobación del índice en
`erasure_steps.last_error`) y `MemoryOrigin` (`fact_sources.origin`). Sin
eso, un estado nuevo de un paso de borrado sobreviviría a un rollback a R1,
cuyo `due()` y cuyo claim sólo buscan `pending`/`running`: el borrado de la
conversación quedaría detenido sin aviso. Cada vocabulario es igual a su enum
en un test, y otro test clasifica cada columna de esas cinco tablas. Las dos
ampliaciones (SC2-R1 y AUTHZ-1) son una enmienda del diseño aceptada el
2026-11-15 como decisión técnica (más estricta que SCv2-P5; revocable hasta
que SC-6a construya R1): se aplicaron antes de R1 porque después ampliar el
contrato exige la época 2, una transición de esquema fuera del modo release.
SC-6a construyó R1 el 2026-11-15 desde 81ed568, así que la época 1 quedó
congelada tal como está: cualquier cambio es época 2.

Quedan fuera los conjuntos cerrados que `src` escribe como literales sin
nombre importable: `capture_runtime_control.mode` (`running`, `muted`,
`resuming`, en un CHECK), `capture_browser_recorders.kind` (tupla en
`capture_recorders.py`), `routine_runs.source`,
`legacy_quarantine_records_v1.state` (CHECK `quarantined`), las claves de
`auth_meta.key` (`schema_version`, `clock_floor`, `legacy_epoch`,
`legacy_digest`; el valor de `schema_version` sí está en el contrato), la
gramática de `reminders.recurrence` y, en las tablas de avance del borrado y atribución,
`policy_index_deletions.state` (`pending`, `running`, `complete` en
`index_deletions.py`), `facts.category` (`personal`, `preference`,
`pattern`, `fact` en `FactLedger.store`) y `voice_groups.state`/`reason`
(literales en `src/users/voice_groups.py`). Agregar un valor ahí no cambia el
pin; quien lo haga debe tratarlo como un cambio de contrato y subir la época a
mano. Tampoco son contrato las columnas de diagnóstico de tablas protegidas,
texto que ningún código usa para elegir filas (los reintentos de borrado
eligen por `attempts`): `reminder_home_actions.reason` (por ejemplo
`interrupted_send`), `routine_runs.reason`, `routine_run_actions.reason`
(`action_exception`, `interrupted`) y `pending_audio_deletions.last_error`
(`live_or_unknown_reference`). Si un cambio vuelve a una de ellas selectora de
un borrado o de autoridad, pasa a ser contrato y sube la época. Un test
clasifica cada columna de texto de cada tabla protegida: vocabulario fijado,
excluida (esta lista), diagnóstico o sin conjunto cerrado. Los enums de las demás tablas no protegidas (por ejemplo
`suggestions_policy_v1.status`) no forman parte del contrato.

Desaparecen con esto el límite de 8 dueños y el hallazgo F8 (un borrado del
head en filas protegidas de un dueño dejaba el head en UNKNOWN): sin dueños
no hay ni uno ni otro. R1 no tiene dueños. SC-5b retiró del código la
constante del límite (`runtime_admission_delegation_owner_limit` ya no existe)
y la rama de concesión heredada del CLI `delegate`, que ahora rechaza siempre
con `runtime_admission_delegation_retired`, también en una raíz de prueba con
`LEGACY_MANAGED_TRANSITIONS`.

Lo que sigue vale sólo para raíces que ya tienen dueños (raíces de prueba y
desarrollo creadas antes de SC-2): la validación de cadenas existentes, la
caché de dueños y la compactación de esos dueños no cambian. Los tests crean
esos mundos con `LEGACY_MANAGED_TRANSITIONS` (fixture
`legacy_managed_transitions` y `tests/mocks/legacy_transitions.py`); un
controlador compilado no tiene forma de activarlo, y los ensayos compilados
construyen dueños con un archivo fijado de época 3 (26de262).

Cada registro de `admission/delegations/` conserva el esquema v1 y nombra al
dueño físico de las rutas. Al lado, `admission/delegation-chains/<capability>.json`
guarda la cadena: dueño primero, emisor al final. El controlador sólo honra
registros con esa cadena; una cadena faltante, alterada o truncada se rechaza
(`runtime_admission_delegation_chain` o
`runtime_admission_delegation_chain_missing`).

La sonda de `activate` también corre el `validate_target_ownership` del
candidato, así que un candidato que recalcularía distinto a un dueño se
rechaza antes de mover el head.

Límites de una raíz con dueños existentes:

- Todo estado dueño debe quedar intacto mientras un descendiente lo use. Si
  cambia su marker o su cierre, si se altera o retira su slot, si se actualiza
  un SDK o cambia el formato del binding, se pierden sus destinos en todos los
  descendientes.
- Costo: cada pasada de validación toma una instantánea completa por dueño
  distinto, no por destino. Un candidato 0b hace lo mismo en su sonda. Un
  candidato anterior a 0b (por ejemplo, una release de rollback sin 0b) toma
  una por destino delegado, en la sonda y en su hijo. Medido en un Mac
  M-series con estados sintéticos de 0,49 GB, un destino con 5 destinos
  heredados de 2 dueños da:
  - sonda 0b: 3,5 s;
  - sonda anterior a 0b: 6,8 s;
  - pasada del controlador: 2,4 s;
  - `activate` completo: 17,5 s.

  El límite de la sonda (60 s por defecto) no cambia. Con 8 dueños (el máximo
  que se concedía antes de SC-2) de ese tamaño, la sonda 0b ronda 10 s. Con estados de unos 2 GB, una sonda 0b con 8
  dueños ronda 40 s. Una anterior a 0b, con varios destinos por dueño, pasa
  de 60 s: hay que medir con una copia real
  (`tests/mocks/probe_cost_harness.py --multi-hop`) antes de la release y
  ajustar `--probe-timeout` sólo con esa medición.
- Sin secreto, quien escribe `admission/` con el mismo uid queda fuera del
  modelo.
- Dentro de un slot 0a, el hijo sólo comprueba el registro del dueño, no la
  cadena.
- Los archivos de historial guardan los registros, no las cadenas.

### Hueco de privacidad conocido: copias que el borrado no alcanza

Estado actual (confirmado por revisión de código, 2026-11-11). Cuando el head gestionado borra una conversación, un recuerdo o un perfil de voz, el borrado actúa sobre:

- su propia base de privacidad (`privacy.db`);
- los destinos con ruta que tiene delegados: eventos, sugerencias, contextos, capturas, diagnósticos, colectores e índices semánticos de los estados dueños anteriores.

La base de privacidad no es un destino de borrado. Ningún participante la nombra (`scripts/stage_runtime.py:1116-1167`). Por eso, en estas copias, las filas de `facts`, `fact_sources`, `fact_dependencies`, `voice_profiles` (huellas biométricas), `voice_fingerprints` y las tablas `voice_attribution_*` **nunca se borran**:

- la `privacy.db` de cada slot gestionado anterior, sea dueño o no;
- los snapshots y estados derivados que registra el catálogo de copias (`admission/copies`, con `acknowledges_retained_copy_cleanup = false`);
- copias que no son miembros del linaje gestionado, aunque algunas estén dentro de `/home/kza/releases`:
  - el estado de producción anterior a la adopción, el del slot `release_20261028T145442Z_b0866e8`, al que enlazan los contenedores de los slots e9669ec (el que sirve hoy) y 63ea8d3. Queda congelado como imagen de rollback (L4) y en él ya no corre la retención: audio ambiente, transcripciones, texto de eventos y recuerdos vencidos siguen ahí. Contiene además `data/audit.db` (si la auditoría estuvo activa) y la tabla heredada `events` de `data/events.db`, con transcripciones y texto personal, que no se llevaron al estado gestionado. Se retira desde DÍA 0 + 14 × 24 h, contado desde el primer cierre limpio gestionado (decisiones P7 = B y Q1 = A; plazo y responsable en «Retiro del estado congelado de b0866e8 (Q5, P7)»);
  - el original anterior al 2026-10-28 en `/home/kza/kza` (`config`, `data`, `.env`). Está congelado y sin retención desde esa fecha, así que guarda más datos vencidos que la imagen anterior;
  - las veinte copias SQLite que el 2026-10-29 se tomaron de ese estado, diez por publicación, en `backups/release_20261029T185008966364Z_e9669ec/databases/` y `backups/release_20261029T174830709997Z_63ea8d3/databases/`. Se retiran las veinte en el mismo paso y con el mismo registro (Q2 = A y R1E-Q1, 2026-11-18);
  - los slots superados `release_20261029T174830709997Z_63ea8d3` (sólo enlaces a ese estado y un runtime) y `release_20261028T144002Z_2c5adb4` (un runtime y una copia más vieja del estado). Se retiran enteros en el mismo paso (R1E-Q2/Q4/Q5);
  - los demás respaldos de `/home/kza/releases/backups/`: `release.conf`, `config` y `.env` del respaldo del 2026-10-29, y el respaldo de `/home/kza/kza` (`release_20261028T144002Z_2c5adb4`). El retiro Q5 no los toca;
  - las métricas anteriores en `/home/kza/logs/` y los logs y la consola anteriores, que pueden contener texto. Los que viven dentro del estado de b0866e8 se van con él; los demás quedan;
  - cualquier snapshot del disco, y los respaldos que contengan los adaptadores LoRA retirados en la ventana por decisión P1 = D.

Los embeddings de esos recuerdos en los índices semánticos de los estados dueños sí se borran, porque esos índices son destinos con ruta.

Al adoptar (U3b-0e), el registro de borrados se siembra con las marcas que ya había en los datos: conversaciones con tombstone, recuerdos en estado borrado, perfiles de voz con `deleted_at` y usuarios retirados. Lo que b0866e8 borró del todo, sin dejar marca, no se puede sembrar. La adopción no guarda nombres ni hashes de los archivos del estado anterior: el plan, el sello y el registro conservan sólo el digest de ese manifiesto (revisión 0e3 F10).

Desde R1 (con U3b-0d4a), la comprobación de resurrección mira cada almacén mapeado por separado y detecta conversaciones, recuerdos y perfiles de voz restaurados. Para conversaciones, los almacenes mapeados son `utterances`, `capture_conversations`, `events_policy_v1`, `wake_clips_v1`, `conversation_turns_v1`, la metadata de Chroma, los JSON de contextos y los JSON/JSONL de colectores (`CONVERSATION_SOURCES` en `scripts/managed_launcher.py`). No detecta una conversación restaurada sólo en lo que `CONVERSATION_NOT_SCANNED` lista con su motivo: el audio de clips y de ambiente (se ve sólo por su fila), la procedencia de voz y de hechos, las sugerencias, los streams, el catálogo de diagnósticos y la cuarentena heredada; eso queda para U3b-0d4b. Los borrados de conversaciones siguen mostrándose «pendiente» hasta U3b-0d4b, porque tres inventarios todavía no tienen limpiador (P8 = A).

Rollback no gestionado: hasta DÍA 0 + 14 × 24 h, contado desde el primer cierre limpio gestionado, sólo al contenedor de e9669ec con decisión registrada y la frase 'revivir datos borrados'. Después del retiro no hay rollback no gestionado: sólo se avanza con una release gestionada, y `/home/kza/kza` no es un destino de rollback (Q3 = A).

Consecuencia: si una persona pide «olvidá mi voz» u «olvidá esto», su perfil de voz o su recuerdo desaparece del estado activo, pero sigue en cada slot anterior y en cada copia hasta que esa copia se destruya. Con la compactación (U3b-0c, desarrollada y no ejecutada; ver «Compactación de estados dueños (U3b-0c)»), esas copias se destruyen al compactar. Mientras tanto pueden ser hasta 5 slots anteriores si se compacta al llegar a 6 dueños. Desde SC-2 no se crean dueños nuevos; el modo release (pendiente) retirará el estado anterior en cada transición. El borrado de la compactación es sólo unlink: no sobrescribe. Las copias fuera de la raíz no las alcanza ningún mecanismo actual.

Hasta que exista una herramienta, retirar a mano una copia vieja requiere una decisión explícita del usuario. Nunca borres el estado del slot activo ni el de un dueño que el head todavía necesita (ver «Todo estado dueño debe quedar intacto»).

Versiones del controlador: una raíz con dueños sólo se valida y se compacta;
no se le agregan dueños. Usá siempre `controller/current` (época 4 o
posterior), nunca el `stage_runtime` de un slot.

### Controlador de época 5 (SC-1a) — desarrollado, sin release

Estado al 2026-11-16: implementado en la rama `sc1a-controller-epoch5`
(diseño `sc-release-design-v4`, unidad SC-1a, aceptado). No hay release:
producción y R1 siguen en época 4. Es la primera pieza del modo release
posterior a R1; plan/start en modo release (SC-1b) está en la sección
siguiente, la corrida de verificación (SC-4a), las salidas (SC-4b) y la
lista cerrada de Chroma (SC-4c) en las suyas y el procedimiento del operador
(SC-5b) en «Procedimiento del modo release (SC-5b)»; falta SC-6b,
así que esta sección no habilita
ninguna transición gestionada (ver «Sin
transiciones gestionadas después de R1 hasta el modo release»). Evidencia:
`data/autonomous-roadmap-20261102/t040-t041/sc1a-evidence.json` (local,
fuera de Git).

Qué es la época 5: sólo modo release para logs nuevos (lo completa SC-1b),
ningún registro de delegación ni dueño se acepta, y `src` capaz de la corrida
de verificación (SC-4a). Una copia de época 4 (R1 incluida) no puede tomar el
lease de una raíz de época 5 (`runtime_admission_controller_downgrade`). Los
slots de R1 siguen sirviéndose: `SCHEMA_CONTRACT_EPOCH` 1, `FORMAT_EPOCH` 1,
`LAUNCHER_CONTRACT` 3 y `RECEIPT_STATE_SCHEMA` 2 no cambian.

Triple del controlador. `release.json` declara `controller_epoch`,
`controller_revision` (`CONTROLLER_REVISION`, hoy 1) y
`controller_sources_sha256`, el sha256 de la lista ordenada (ruta, sha256 del
fuente) de la clausura de imports del controlador, que `build_runtime`
calcula y también registra (`controller_sources`). La clausura sale de un
análisis estático de todos los `import` de `src.*` y `scripts.*` (de módulo y
de función) desde `stage_runtime`, `compaction`, `managed_launcher`,
`adoption_commands` y el worker `src/core/compaction_worker.py`, más los
módulos que el controlador carga por ruta y `src/memory/copy_observation.py`,
cuyo formato lee con su propio parser. Hoy son unas 300 fuentes, casi todo
`src`: casi cualquier release cambia el triple, y entonces necesita un T0 con
`CONTROLLER_REVISION` subido. El triple instalado se lee de
`controller/current/release.json` (la copia que hace la instalación);
`admission/controller.json` conserva sus cinco campos.

`install-controller` desde la época 5, siempre antes de mover `current` y sin
escribir nada en los rechazos:

- se corre desde el archivo NUEVO extraído, nunca desde el `$ST` instalado:
  cada módulo de la clausura del árbol que corre la instalación debe ser el
  del archivo (`runtime_admission_controller_installer_mismatch`); compilado
  se comparan los `.pyc` con los del archivo, y desde un checkout de fuentes
  los `.py` con `controller_sources`, así que un T0 desde otro checkout
  también rechaza;
- monotonía en (época, revisión) contra el triple de
  `controller/current/release.json`, que se lee siempre que exista
  `controller/current`, aunque `controller.json` falte o esté dañado: menor
  rechaza (`runtime_admission_controller_downgrade`); el mismo par con otro
  digest rechaza (`runtime_admission_controller_revision_conflict`); el mismo
  triple vuelve a correr todas las comprobaciones y no cambia el
  controlador, pero un archivo de otro commit con la misma clausura sí mueve
  `current` a ese commit y reescribe `controller.json` y el historial. Si
  `controller/current/release.json` no se puede leer, toda instalación
  rechaza (`runtime_admission_controller_tampered`): sólo se repara a mano
  (trabajo de desarrollo, servicio caído, decisión escrita cada 48 horas);
- primera instalación de época 5 (el `controller.json` instalado declara
  época menor que 5, falta o no se puede leer): rechaza un log de compactación pendiente
  (`runtime_admission_compaction_pending`) o ilegible
  (`runtime_admission_compaction_log_invalid`), un head abierto
  (`runtime_admission_not_closed`), cualquier entrada en
  `admission/delegations/` (`runtime_admission_delegation_frozen`) y un head
  con dueños (`runtime_admission_owners_present`); después corre SR-1;
- SR-1 sobre el head cerrado: el binding del slot recalculado por el código
  nuevo debe ser idéntico al guardado
  (`runtime_admission_controller_binding_changed`; si sólo difiere
  `sdk_versions`, `runtime_admission_compaction_sdk_changed: <paquete>==<versión
  ligada>`: reinstalar esa versión), la sonda del propio slot, el mismo
  contrato de esquema que el slot (`runtime_admission_controller_schema_contract`)
  y las comprobaciones de slot de `serve`;
- instalaciones siguientes: leen el log pendiente por su cuenta; uno
  ilegible rechaza (`runtime_admission_compaction_log_invalid`). Con el head
  cerrado (o sin head) sólo se admite sin log o con un log del modo release,
  y un log simple pendiente rechaza (`runtime_admission_compaction_pending`):
  primero se termina con los comandos simples; después, SR-1. Con un log
  del modo release y un head cerrado que no es H (K', filas A0, V3, V4 y SS),
  además el binding de H recalculado debe ser igual al del log
  (`runtime_admission_controller_binding_changed`), salvo que el log ya
  tenga `retire_started`: ahí H se está destruyendo y no se comprueba.
  Un head abierto
  sólo entra bajo un log pendiente del modo release y sólo si es el H o el K
  autorizado de ese log; entonces el binding de H recalculado debe ser igual
  al del log. Cualquier otro head abierto es desconocido y rechaza
  `runtime_admission_not_closed`: cerrarlo antes con R-b o `adopt`.

Corte entre el cambio de `current` y `controller.json` (fila T0S, en la
primera instalación o en una siguiente): todo lo que no sea
`install-controller` rechaza con
`runtime_admission_controller_install_incomplete` (`serve` incluido) hasta
repetir la instalación del mismo archivo, que vuelve a correr las mismas
comprobaciones (un archivo menor que el que quedó en `current` rechaza por
monotonía). Un corte después de `controller.json` y antes del registro
inmutable del historial, repetido, pierde la entrada del historial que
nombraba el controlador anterior. Un
corte antes del cambio (fila T0) deja el controlador anterior: H sigue
sirviendo con él.

Guarda AUTHZ3-1 (sin estado): en época 5 cualquier entrada en
`admission/delegations/` hace rechazar a `serve`, `plan` y toda validación de
binding (`runtime_admission_delegation_frozen`), y un head cuya política lista
destinos delegados no se sirve (`runtime_admission_owners_present`) ni se
planifica (`runtime_admission_compaction_owners_present`). Un registro que
aparece después del T0 significa que corrió una copia vieja (fila DL): si el
head no tiene dueños, borrarlo a mano restituye `serve`. `compact abort` (en
todos los modos) y `compact resume` de un abort comprueban la guarda antes de
escribir cualquier evento: con un registro presente rechazan
`runtime_admission_delegation_frozen` sin escribir nada, y la salida de la
fila DL (borrar el registro y repetir el mismo comando) vale igual con un log
pendiente. Si además hay dueños
en el head (fila DO), no borrarlo: la salida es L4 mientras exista y sin log
pendiente, o trabajo de desarrollo. `uninstall-launcher` (L4) no queda
bloqueado por la guarda. Es detección, no prevención: una copia vieja todavía
puede mover el head.

Instalador equivocado: si el primer T0 se corre con el `$ST` de época 4, no
corre ninguna de esas comprobaciones, y las guardas de `plan` y `serve` siguen
vigentes. Un log simple pendiente **sin dueños** queda arrastrado y lo
terminan los comandos simples de la época 5; después, repetir la instalación
con el instalador correcto es el mismo triple y deja admitido el plan
(mientras el log siga pendiente, esa repetición rechaza
`runtime_admission_compaction_pending`). Un log de compactación de estados
dueños pendiente **no tiene salida**: `compact verify` sólo arma, `serve`
sale con 78 (`runtime_admission_delegation_frozen`), `compact abort` y
`compact abort --to-h` rechazan con la misma guarda antes de registrar nada
(`runtime_admission_delegation_frozen`; nunca queda `abort_started`), L4
rechaza por el log pendiente y un
controlador corregido de época 5 rechaza por los registros. Es una parada
terminal (trabajo de desarrollo, servicio caído, decisión escrita cada 48
horas). Precondición del T0: ninguna compactación de dueños pendiente; si
la hay, se retira (o se aborta) en la época 4 antes del T0, y un controlador
corregido para terminarla debe ser una construcción de época 4.

Tests: los mundos con dueños (compactación simple de estados dueños,
cadenas de delegación) corren con una construcción de época 4 de este mismo
árbol (`tests/mocks/epoch4.py`); los rechazos de época 5 y las filas T0, T0S,
T0F, DL y DO están en `tests/unit/tools/test_sc1a_controller_install.py`.

### Plan y start del modo release (SC-1b) — desarrollado, sin release

Estado al 2026-11-16: implementado en la rama `sc1b-release-plan-start`
(diseño `sc-release-design-v4`, unidad SC-1b, aceptado). No hay release. Con
SC-1b una transición del modo release se planifica y se empieza (K activado
con el código NUEVO); SC-4a y SC-4b agregan la verificación y las salidas en
desarrollo, sin release (ver «Sin transiciones gestionadas después de R1 hasta
el modo release»). Evidencia: `data/autonomous-roadmap-20261102/t040-t041/sc1b-evidence.json`
(local, fuera de Git).

Modo: con el controlador en época 5 todo log nuevo es del modo release
(`intent.mode = "release"`, la clave que ya lee `install-controller`). Un log
sin `mode` (simple, arrastrado de la época 4) sólo lo terminan los comandos
simples. Una raíz con dueños rechaza en `plan` (SC-1a). Los comandos siguen
corriendo desde `controller/current`; `--archive` es el archivo NUEVO.

`compact plan <K> --archive NUEVO` (P6'), de sólo lectura:

- (a) el `release.json` del NUEVO declara `launcher_contract` >= 3
  (`runtime_admission_launcher_contract_missing`) y `format_epoch` >= 1
  (`runtime_admission_rollback_format_closed`);
- (b) la época del controlador del NUEVO está entre 5 y la instalada: menor
  que 5, `runtime_admission_compaction_release_epoch` (R1, de época 4, nunca
  es destino, SC-D7 A); mayor que la instalada (o `controller/current` aún
  en R1), `runtime_admission_compaction_t0_required` con los dos triples: hace
  falta el T0 del NUEVO;
- (c) el NUEVO está en served(root) o su triple es el instalado. served(root)
  es el archivo de cada log del modo release terminado (`complete`) con
  `verified`, más la release del head actual; se deriva de los logs
  inmutables, sin registro de historial. Un log simple, uno terminado sin
  `verified`, uno abortado o uno pendiente no sirve nada. Si no: triple mayor
  que el instalado, `runtime_admission_compaction_t0_required` (hace falta el
  T0 del NUEVO); menor y no servido,
  `runtime_admission_compaction_controller_behind` (un arreglo hecho desde una
  línea de releases vieja: rehacerlo sobre la línea del controlador
  instalado); misma (época, revisión) con otra clausura,
  `runtime_admission_compaction_controller_revision_conflict`. Los tres nombran
  `installed=<época>.<revisión>.<digest>,archive=<época>.<revisión>.<digest>`.
  Una vuelta atrás a una release ya aceptada no necesita T0;
- (d) H pasa las comprobaciones de slot de `serve`;
- (g) SR-2: el contrato de esquema del NUEVO, el del controlador instalado y
  el del slot de H son iguales y >= 1
  (`runtime_admission_compaction_schema_contract`; SC-D4 A);
- (e) el digest del plan, y después el intent, ligan `mode`, `release_sha256`,
  `git_commit`, `schema_contract` y el triple del NUEVO.

Además, en el modo release `--retire-prepare-leftovers` queda forzado, y
cualquier entrada intacta bajo `slots/` rechaza
(`runtime_admission_compaction_release_unproven_copy`, con las rutas y sin
contenido), salvo los dos slots heredados que nombra el registro de adopción
confirmado (el contenedor de e9669ec y el slot de b0866e8; con dos registros
confirmados, o uno cuyo nombre no es el digest de su head, no se exime
nada). En producción hay dos slots legados más, `release_20261029T174830709997Z_63ea8d3`
(sólo enlaces) y `release_20261028T144002Z_2c5adb4` (una copia completa
anterior). Antes del día 14 el plan los rechazaría; el retiro Q5 del día 14
los borra (decisión R1E-Q2/Q4/Q5 del 2026-11-18), así que el primer plan del
modo release, que va después, ya no los ve (probado sobre la forma de
producción en `test_r1e2_after_day_14_r2_s_first_release_plan_finds_no_unproven_copy`). Un plan rechazado no escribe nada (fila P): arreglar la causa y
repetir, o habilitar y arrancar la unidad, que sirve H. Qué hacer con cada
ruta rechazada: un `slots/.prepare-*` vacío (un `start` cortado entre crear
el directorio de staging y escribir su registro) se quita con `rmdir` y se
repite el plan; cualquier otra entrada no se borra: el servicio queda en H,
se averigua qué es y se registra la decisión de 48 horas (runbook de
compactación, tabla de códigos).

`compact start`: K se prepara desde el archivo NUEVO y `_k_runtime` exige
que su `release.json` sea el del intent (P6'(f)). `compact resume` no repite
la admisión: antes de extraer nada compara el `release.json` del archivo dado
con el del intent y rechaza `runtime_admission_compaction_artifact`
(`"archive": "not_the_intent_archive"`) sin escribir, así que un `resume` con
el archivo de H después de un `start` cortado no deja un K equivocado y el
`resume` con el NUEVO sigue funcionando; A1 exige además que los
`schemas` del binding de K sean los de H
(`runtime_admission_compaction_schema_changed`); A2 sigue probando K con su
propio código antes del CAS. `allow-run`, `--discard-k-writes` y la vuelta a
H por Q6 rechazan `runtime_admission_compaction_plain_mode_only` (la vuelta a
H del modo release es W, de SC-4b). SC-4a quitó la negativa de SC1B-N3
(`release_verify_unavailable`) y la reemplazó por una compuerta de release
(SC4A-N13) que SC-4b levantó: `start`/`resume` dicen `next: compact verify
<id>` también para un log del modo release, y antes de armar V el abort
simple sigue sin pérdida (fila A0). Un `intent.mode` distinto de
`"release"` no es un log simple: `verify`, `allow-run`, `--discard-k-writes`,
la vuelta a H por Q6, `plan` (served(root)) e `install-controller` lo
rechazan como `runtime_admission_compaction_log_invalid`, `compact status` lo
lista como inválido, y el abort simple antes de la corrida sigue disponible.

Reubicación de las colas de audio (toda compactación, también la simple):
`pending_audio_deletions.path` y `audio_write_intents.path` bajo H (o un
dueño) se reubican en K como `utterances.audio_path`, y A1 compara la
proyección protegida de K con la de H con esas rutas reubicadas. Antes, una
cola no vacía bloqueaba toda compactación
(`runtime_admission_compaction_absolute_locator:...:pending_audio_deletions.path`)
sin dejar nada viejo. Un valor que nombra H sin estar debajo sigue rechazando,
y dos filas que caerían en la misma ruta de K rechazan
(`runtime_admission_compaction_locator_collision:<tabla>.path`) antes de
escribir.

`--accept-regressed-source` (R5-1) y el testigo de cierre por registro de
recuperación, que esta unidad dejó en `follow_up`, están implementados desde
la unidad SC-PA-DR (fila PA, sección «Filas PA y DR (SC-PA-DR)»). Queda
pendiente (lista `follow_up` de la evidencia): tampoco se exige todavía que
el código corra desde `controller/current` (revisión RA-6): P7 acepta
cualquier `controller/<commit>` de la misma época, así que después de un T0
que sube la revisión un `controller/<commit>` anterior podría planificar y
empezar con su propio trabajador. No afecta a la primera actualización
posterior a R1 (el único otro directorio es el de R1, de época 4, que no
toma el lease de una raíz de época 5), pero la regla del operador sigue
siendo usar sólo `controller/current`.

Tests: `tests/unit/tools/test_sc1b_release_plan.py`.

### Corrida de verificación aislada del modo release (SC-4a) — desarrollada, sin release

SC-4a implementa V del diseño v4 (`mechanism.v_verification_run`, enmiendas
A1/A2; decisiones del usuario SC-D5 = A, corrida aislada sin tráfico, y SC-D8
= A). Nada de esto está instalado: el servidor sigue en R1 (época 4) y
ninguna transición gestionada existe hasta cerrar el resto del modo release
(sección siguiente).

- **Bandera.** `compact verify <id> --smoke-seconds N` arma la única corrida
  del log (modo simple: 10..3600, 120 por omisión; modo release: N explícito
  de 10 a 60, ver más abajo; la compuerta SC4A-N13 la levantó SC-4b).
  Para esa corrida, y sólo
  para ella, `serve` agrega `KZA_RUNTIME_RELEASE_VERIFY=1` al entorno del
  hijo después del filtro que quita todo `KZA_RUNTIME_*` de la unidad: la
  unidad no puede inyectarla, y ni H ni las corridas de un log abortado o
  completo la llevan. El hijo la lee una vez (`take_release_verify_flag`), la
  quita del entorno y la exige junto al ticket.
- **NUEVO en modo verificación.** `main.build_application` inyecta
  `verify_only` por constructor en el grafo, la autoridad del tablero, los
  servicios de privacidad, el ledger y los constructores con mantenimiento.
  El grafo construye, abre, valida y carga, pero no arranca ningún bucle, no
  corre ninguna reparación de una vez y la autoridad termina toda transacción
  con ROLLBACK (también la rotación final del cierre). La lista es la medida
  por SC-0 (`skip_list`): poda de dueños retirados al arrancar (salteada, no
  diferida), bucle de privacidad, `erasures.recover` e
  `index_deletions.recover`, `register_target` sin INSERT (cubre los tres
  sitios), poda de `fact_owners` al abrir, `active -> unknown` de las
  grabadoras, `sending -> uncertain` de recordatorios, reparación y bucle de
  diagnósticos, retención de captura, registro de voces (poda y bucle), clips
  de wake (bucle; el borrador de colectores sólo corre desde el coordinador,
  que no arranca), recortes de retención de eventos y sugerencias al abrir,
  sus barridos, y los bucles de índice, coordinador de borrado, bus en vivo,
  memoria y contextos. `Application.run` toma la rama de verificación: no
  corre `_start_services` (sincronía de HA, rutinas y su
  `recover_interrupted`, presencia, recordatorios, entrenador nocturno), ni el
  tablero, ni el camino ambiental, ni el audio; espera
  `VoicePipeline.prepare()` (modelos, índice y memoria), envía la línea
  privada `VERIFY=<run_id>` y después `READY=1`.
- **Cierre de captura.** En V ningún navegador puede llegar, así que el
  `drain` de las grabadoras no borra las reservadas ni espera confirmaciones,
  y el cierre exige las filas `(token_hash, state, kind)` iguales a las
  leídas al construir; la prueba de cierre cuenta las filas llevadas
  (`capture_permissions_carried`, un número).
- **Registros de `serve`.** `admission/runs/<run_id>.ready.json` se escribe
  sólo si `VERIFY` con el `run_id` del ticket llegó antes que `READY=1`; desde
  ahí `serve` mide N segundos con el reloj monotónico, escribe
  `<run_id>.smoke-stop.json` (segundos medidos y motivo) y detiene él mismo al
  hijo, que sella con su recibo (salida 0, sin reinicio). `READY` sin
  `VERIFY` (un NUEVO que ignora la bandera), con otro `run_id` o después de
  `READY` detiene al hijo enseguida, no escribe `ready.json` y deja
  `verify_unacknowledged` (fila V3). Una parada externa antes de N deja el
  intervalo corto (fila V3).
- **Verificada.** En el modo release `verified` exige además los dos registros
  y `measured_seconds >= N` monotónicos; un salto del reloj de pared no cuenta.
  Una corrida sellada sin ellos no se verifica y `serve` la rechaza con
  `runtime_admission_compaction_verify_consumed`.
- **Estado.** `compact status` muestra `verify_branch_reached`: `yes` sólo
  prueba que el código NUEVO llegó a su rama de verificación, no que V
  estuvo aislada ni que nadie usó el sistema; `no` quiere decir que ni eso
  quedó probado; `not run`, que la corrida armada no empezó; `unreadable`,
  que un registro de la corrida no se puede leer (el estado sigue mostrando
  la fase del log en vez de `invalid`).
- **Qué quiere decir «aislada».** Sin entrada (ni micrófono, ni tablero, ni
  camino ambiental) y sin efectos en la casa. No quiere decir sin red: como
  cualquier arranque, V hace al construir las mismas pruebas de conexión sin
  contenido a Home Assistant (`test_connection`), al gateway LLM `:8200`
  (`HttpReasoner.load`) y al router local `:8101`.
- **N del modo release (SC4A-N14).** N es un término del presupuesto de
  SC-D2 A (como mucho 10 minutos de servicio caído): `formula_check` del
  diseño dice N = min(60, 600 − T_copy − arranques × B − paradas), nunca
  menos de 10. Por eso un log del modo release no tiene N por omisión
  (`compact verify` sin `--smoke-seconds` rechaza
  `runtime_admission_compaction_release_smoke_seconds`) y rechaza N > 60.
  Con los valores de hoy (T_copy ≈ 146 s, B ≈ 146 s de la compuerta de W8):
  primera transición, con T0, N ≈ 10 s; transiciones posteriores, N ≤ 60 s.
  SC-6b fija el valor por omisión con los B y T_copy medidos.

**Compuerta de release hasta SC-4b (SC4A-N13, revisión RA-1/RB-1), levantada
por SC-4b.** Sin W, después de la corrida no había salida por la herramienta
(SC4A-N1: el retiro simple rechazaba un log del modo release y Q6 y el abort
simple también), así que `RELEASE_EXITS_AVAILABLE` (en
`scripts/compaction.py`) impedía armar V. SC-4b agrega W, el abort con W y el
retiro con igualdad W (sección siguiente) y la pone en `True`: `compact
verify` arma un log del modo release (con N explícito, SC4A-N14) y
`start`/`resume` dicen `next: compact verify <id>`. El mecanismo queda y una
prueba lo fija con la constante en `False`. Sigue en `follow_up`: el recibo no
registra todavía `capture_permissions_carried` (cambiaría el esquema del
recibo que el controlador compara exacto).

Tests: `tests/unit/tools/test_sc4a_verify_run.py`,
`tests/integration/test_sc4a_verify_boot.py`,
`tests/unit/memory/test_sc4a_verify_stores.py`,
`tests/unit/core/test_sc4a_application_verify.py`; arnés de SC-0 con la lista
activa: `python tools/sc0/run_states.py --verify --revision <commit>
--template-revision <commit de H>` (por ejemplo, H de R1 con
`--template-revision 357dcb5`; cada resultado nombra los dos commits).

### Salidas de V del modo release: W, abort y retiro (SC-4b) — desarrolladas, sin release

SC-4b implementa las salidas de la corrida V del diseño v4
(`mechanism.w_abort_predicate`, `retire`, `safe_stop`, filas de
`crash_table`; decisiones del usuario SC-D6 = A, SC-D7 = A, SC-D8 = A). Nada
está instalado: el servidor sigue en R1 (época 4) y el hueco de la sección
siguiente sigue abierto hasta SC-6b (SC-4c, SC-R y el runbook de SC-5b ya
están).

- **W0 (H intacto).** Sin hijo vivo de V, sin escritores abiertos en K' y H
  igual a su binding, marcador y prueba de cierre. Lo que se cura solo es
  transitorio (fila AT, no escribe nada, se repite después):
  `runtime_admission_writer_active`, `runtime_admission_busy`,
  `runtime_admission_sdk_version_unavailable`,
  `runtime_admission_compaction_sdk_changed` (el binding difiere sólo en
  `sdk_versions`: es el venv compartido, no los archivos de H; el informe
  nombra la versión ligada y la instalada, y la salida es reinstalar la
  ligada) y `runtime_admission_compaction_h_transient` (errno EMFILE, ENFILE,
  EIO, EAGAIN, EINTR, ENOMEM, EBUSY; SQLite BUSY, LOCKED, IOERR, NOMEM,
  INTERRUPT, por código y no por clase). Cualquier otra falla al leer H
  (ENOENT, otro código SQLite, binding distinto, marcador, prueba de cierre)
  es `runtime_admission_compaction_h_changed` con el código de abajo: la
  parada terminal HD. Revisión 1 (RB-1): una entrada de lock de K' que no se
  puede abrir (un enlace simbólico, sin permiso) no es un error sin código:
  W0 la informa como `k_writers: unverifiable` con su errno y W1 da
  `restrictive_or_unclear` (SS y su salida); un errno de recursos ahí
  (EMFILE, ENFILE, EAGAIN, EINTR, ENOMEM, EBUSY) es el transitorio
  `runtime_admission_compaction_k_transient`.
- **W1 (K' contra K).** Abre cada base que nombra la proyección autorizada de
  K con `mode=rw` (recupera en el lugar un journal caliente o un WAL, nunca
  crea el archivo) y compara las mismas proyecciones que
  `capture_state_authority`, sin `_schema_contract`; la selección es una sola
  función compartida (`_authority_selection`, digest dorado fijado en
  `test_sc4b_w.py`). Lo que sólo existe en K' se ignora. Una base que falta o
  no se lee cuenta como todas sus filas ausentes; un valor que la selección no
  clasifica cuenta como fila cambiada. El informe no tiene contenido: por
  tabla, filas agregadas y quitadas; por ledger, `equal`, `strict_prefix` u
  `other`; por base, `ok`, `missing` o `unreadable` (con el código); cuentas y
  configuración; y el encabezado `verify_branch_reached`. No lee el reloj: el
  mismo K' y el mismo H dan el mismo digest. Revisión 1: W1 nunca lanza un
  error sin código (RA4B-1: un valor de otro tipo, por ejemplo un
  `sources_json` nulo o un `expires_at` de texto, cuenta como fila sin
  clasificar; cualquier otra falla al leer una base de K' es `unreadable`
  con un código sin contenido); abre una base, un registro de cuentas o un
  archivo de configuración de K' sólo si es un archivo regular en su propia
  ruta resuelta, antes de la apertura `mode=rw` (RA4B-4: un directorio de K'
  enlazado a H nunca se escribe); y un lock SQLite de otro proceso sobre una
  base de K' (SQLITE_BUSY, LOCKED, NOMEM, INTERRUPT) o un errno de recursos
  rechaza el transitorio `runtime_admission_compaction_k_transient`, que no
  escribe nada (RA4B-5/RB-7). EIO, SQLITE_IOERR y el daño siguen como
  `unreadable`: una falla que persiste no puede cerrar la única salida de SS
  (SC4B-N19).
- **Clases.** `equal`; `loss_only` (filas quitadas de `W_LOSS` =
  conversation_tombstones, conversation_erasures, retired_users,
  fact_owner_revisions, voice_attribution_suspensions,
  pending_audio_deletions; filas agregadas a `W_GRANT` = credentials,
  sessions; un ledger que es prefijo estricto del de H): volver a H restaura
  todo eso y no revive nada; `restrictive_or_unclear`: todo lo demás, y
  también una base que falta o no se lee (SC4B-N3), cuentas o configuración
  distintas, y un valor sin clasificar. Una fila cambiada aparece como una
  quitada y una agregada (los hashes no se emparejan), lo que da la misma
  clase que el diseño.
- **Abort.** `compact abort <id> --to-h --confirm-h <prefijo>` de un log del
  modo release usa W en lugar de Q6: con `equal` o `loss_only` vuelve a H
  (`abort_started` registra la clase y el digest); con
  `restrictive_or_unclear` rechaza
  `runtime_admission_compaction_abort_protected_changed` con el informe y su
  digest y no escribe nada: la parada segura SS. Única salida de SS (SC-D8 A):
  una persona revisa el informe y, si confirma por conocimiento directo que
  nadie usó el sistema durante V, corre lo mismo con `--discard-k-changes
  <12+ hex del digest del último rechazo>`; se recalcula W, el digest tiene
  que coincidir y `abort_started` guarda el digest y las cuentas. Un digest
  equivocado, o un descarte sin nada que descartar, rechaza
  `runtime_admission_compaction_discard_k_changes_mismatch` (SC4B-N6). Si el
  head se movió después de `abort_started`, `resume` y el abort simple
  rechazan `abort_recheck_required` y `abort --to-h` rehace W (con su propio
  digest para el descarte, guardado en `abort_rechecked`). El abort simple
  después de V sigue rechazando `abort_after_run`; `--discard-k-writes` es
  sólo del modo simple, y `--discard-k-changes` sólo del modo release
  (`runtime_admission_compaction_release_mode_only`). Un head cerrado cuya
  prueba de cierre se lee pero nombra otra corrida rechaza
  `runtime_admission_compaction_head_moved`; una prueba de cierre de K' que
  no se puede leer rechaza `runtime_admission_compaction_k_closure_unreadable`
  (revisión RB-3), también en el retiro. Las dos son paradas como HD: sólo
  aparecen con daño o una edición bajo `admission/`, o con una corrida de K
  detrás de la compuerta (un controlador viejo), y no tienen salida por
  herramienta.
- **Retiro.** `compact retire <id>` de un log del modo release exige, antes
  de escribir nada más que el `verified` de siempre: binding de K' igual (si
  difiere sólo en `sdk_versions`, `sdk_changed` transitorio; si difiere en
  otra cosa, `runtime_admission_compaction_k_binding_changed` y la salida es
  el abort), prueba de cierre (`verify_closed_binding`), W0 y W1 `equal` sin
  tolerancias, y además el digest de toda la autoridad de K' igual al
  autorizado (SC4B-N4: también rechaza objetos protegidos que sólo existen en
  K' o otro contrato de esquema), y ningún id resucitado en K'
  (`runtime_admission_compaction_retire_resurrected`, cuentas sin contenido).
  Un retiro rechazado por W muestra el informe; la salida es el abort con W.
- **HD.** Con H cambiado, `abort --to-h` (con o sin descarte), el abort
  simple, el retiro, `adopt` y el `uninstall-launcher` de L4 rechazan y no
  escriben nada. No hay salida por herramienta: unidad deshabilitada, nada
  cambiado bajo la raíz ni en `/home/kza/kza/.venv`, y desarrollo con las
  decisiones escritas cada 48 h (runbook).
- **Adopt.** Con un log del modo release pendiente, `adopt` y `adopt
  --dry-run` rechazan `runtime_admission_compaction_pending` antes de
  cualquier copia (R5-PRIV-3).
- **Estado.** `compact status` no calcula W: muestra `armed_run` (`not
  armed`, `not consumed` o `consumed`), `verified` y `verify_branch_reached`,
  sin escribir.

Pendiente, no resuelto: el diseño marca toda la unidad SC-4b como previa a
la primera transición gestionada (`before_first_managed_transition`). Las
partes que esta implementación dejó en `follow_up` siguen siendo previas a
esa transición salvo lo que el usuario respondió el 2026-11-16 a SC4B-Q1 y
SC4B-Q2 (`t040-t041/user-decisions-20261110.json`, clave
`sc4b_questions_20261116`): la primera transición puede hacerse sin
`closed_witness` con registros de recuperación ni `--accept-regressed-source`
si cae dentro de los 14 días de L4 (si cae después, se implementan antes), y
el retiro con escaneo de resurrección absoluto se acepta si el conteo de solo
lectura del paso 0 del runbook da 0 (si no, se implementan antes la línea
base salada de K y el reencolado). `closed_witness` con registros de
recuperación, `--accept-regressed-source` (fila PA) y la sombra de `adopt
--dry-run` bajo `admission/dry-run/` (fila DR) están implementados desde la
unidad SC-PA-DR (sección «Filas PA y DR (SC-PA-DR)»). Lo que queda en
`follow_up`: la línea base salada de resurrección de K y el reencolado del
residuo no protegido (hasta entonces el escaneo del retiro es absoluto: un H
con residuo hace rechazar el retiro, y el abort vuelve a H sin pérdida;
SC4B-Q2). El tiempo
del retiro y del abort con W (T_retire, T_abort: binding de K', W0 y W1 con
imágenes en memoria, escaneo de resurrección, destrucción) no está en la
fórmula D de SC-D2 ni se midió: queda en la lista de medición de SC-6b
(revisión RB-5).

Tests: `tests/unit/tools/test_sc4b_w.py`,
`tests/unit/tools/test_sc4b_release_exits.py`.

### Lista cerrada de colecciones de Chroma en la compactación (SC-4c) — desarrollada, sin release

SC-4c implementa `mechanism.start.chroma_allowlist` del diseño v4 (PRIV-3,
PA-8, V4R2-C1). Vale para toda compactación (modo release y modo simple),
porque corre en el trabajador del controlador. Antes, la copia era idéntica
también para Chroma, y un `data/chroma_db` (el índice de comandos y las
rutinas, que no es destino `semantic_index` registrado) hacía rechazar el
snapshot con `state_snapshot_unregistered_sdk_root`: en producción ninguna
compactación podía pasar del snapshot.

- **Qué recibe K.** Cada almacén de Chroma de H (cada directorio con un
  `chroma.sqlite3`) se reconstruye en K con el SDK público desde la
  exportación del snapshot; nunca se copian sus bytes, así que lo que queda
  afuera no sobrevive en páginas libres ni en los archivos del índice HNSW.
  (1) La colección de memoria de política (la que H registró como destino
  `semantic_index` y sólo con nombre `<colección>_policy_v1`) recibe sólo los registros de
  hechos que `get()` devolvería en t_clone (estado `automatic` o
  `confirmed`, sin vencer, sin conversación fuente con tombstone) cuyo dueño
  no está en `retired_users`, con el registro de borrados como segunda
  compuerta, y sólo si el registro indexa esa revisión exacta (documento
  igual al contenido, metadata `{fact_policy: 1, revision}`). Quedan afuera,
  por ejemplo, un hecho vencido que el barrido todavía no limpió, un hecho de
  un dueño retirado cuya poda no corrió y un id huérfano. Las filas de
  `facts` siguen idénticas (lo que vence lo vence K en su corrida). (2) Las
  colecciones del índice de comandos (`home_assistant_commands`, su
  `__pointer` y las versionadas `home_assistant_commands__<AAAAmmddHHMMSS><6
  hex>`, exactamente los nombres que crea `ChromaSync`) se copian
  enteras: registros, metadata (el puntero conserva `active`) y
  configuración. (3) `home_assistant_routines` de `data/chroma_db` **no se
  copia** (texto de rutinas dictadas, sin camino de borrado); el arranque de
  K la recrea vacía (`ChromaSync._construct`). Es una baja con nombre exacto
  y sólo en ese almacén (respuesta SC4C-Q6 del usuario, 2026-11-17: el par
  `("data/chroma_db", "home_assistant_routines")` de
  `CHROMA_DROPPED_IN_STORE`, igual en el plan y en el trabajador y fijado
  por un test a `vectordb.path` y `vectordb.routines_collection` de
  `config/settings.yaml`, al valor por defecto de `routines_collection` de
  `ChromaSync` y a la ruta por defecto de `src/main.py`); el mismo nombre en
  cualquier otro almacén rechaza como en (4). (3b) La colección heredada `user_memories` de
  `data/memory_db` (anterior a la política de memoria, en cuarentena y
  nunca leída: `long_term.get_stats` informa `legacy_policy:
  "quarantined"`) **tampoco se copia** (respuesta SC4C-Q2b del usuario,
  2026-11-16; producción tiene 876 registros ahí junto a los 130 de
  `user_memories_policy_v1`). Es una baja con nombre exacto y sólo en ese
  almacén (`CHROMA_DROPPED_IN_STORE`, igual en el plan y en el trabajador,
  fijada además a `memory.chroma_path` y `memory.long_term_collection` de
  `config/settings.yaml`): el snapshot no la lee, el hijo no la recibe y
  queda registrada sólo por nombre. Sus bytes siguen en H hasta que el
  retiro lo destruye; ahí se borran para siempre. El snapshot
  (`<raíz>/compaction/<id>/snapshot-n`) también guarda una copia opaca de
  `data/memory_db/chroma.sqlite3` y, en producción, de los archivos HNSW y
  `index_metadata.pickle` de esa colección, con sus bytes; el retiro y el
  `abort` borran el snapshot y con él esa copia (revisión SC4C-RV2-2).
  (4) Cualquier otra colección, incluidos un nombre parecido (`user_memories_v0`,
  `User_memories`), `user_memories` fuera de `data/memory_db`,
  `home_assistant_routines` fuera de `data/chroma_db` (SC4C-Q6) o una
  `user_memories` registrada como destino, rechaza
  `runtime_admission_compaction_unlisted_collection`: en el `plan` (lee los
  nombres de colección de una imagen privada de cada `chroma.sqlite3`, nunca
  un registro; el informe nombra almacén y colección) y otra vez en el
  trabajador (en el snapshot, antes de exportar nada, y en el plan del
  clon). Un rechazo del trabajador es la fila S: `compact abort <id>` (sin
  pérdida, antes de V) y la unidad vuelve a servir H.
- **Proceso hijo limpio.** Cada almacén lo escribe un intérprete nuevo
  (`python -I -B`, sin importar `src`) que recibe por stdin sólo lo que K
  conserva. Motivo, medido en el server: Chroma guarda buffers del índice
  HNSW cuya capacidad sin usar puede tener contenido anterior del heap del
  proceso; escrito desde el trabajador (cuyo heap tiene todo el snapshot de
  H), un `data_level0.bin` de K llegó a tener el texto de un hecho vencido.
  Un fallo del hijo rechaza `runtime_admission_compaction_chroma_rebuild_failed`
  sin devolver su stderr.
- **El hijo no sobrevive al trabajador como escritor (revisión SC4C-RV-1).**
  La revisión mostró en el server que, si el trabajador moría con SIGKILL
  (el OOM killer, un `kill -9`) durante la reconstrucción, el hijo seguía y
  volvía a crear el almacén de K, con los hechos, después de que `abort`
  lo había borrado. Ahora: (a) en Linux el hijo pide `PR_SET_PDEATHSIG`
  (SIGKILL) antes de leer nada, así el kernel lo mata con el trabajador; en
  toda plataforma, antes de cada escritura sale si su padre ya no es el
  trabajador; (b) hereda el descriptor del lease de admisión del
  trabajador, así que mientras vive el lock sigue tomado y `abort`,
  `resume`, `plan` y cualquier otra entrada rechazan `runtime_admission_busy`
  (lo impone el kernel; no hace falta registrar su pid ni su hora de
  arranque); (c) una reconstrucción de un almacén que tarda más que
  `CHROMA_REBUILD_TIMEOUT` (600 s, el presupuesto de bajada entero de
  SC-D2 A) mata al hijo y rechaza
  `runtime_admission_compaction_chroma_rebuild_timeout`.
- **D5 para almacenes reconstruidos.** El hijo vuelve a leer su almacén y
  el sha256 de esa imagen (exactamente las colecciones planeadas, cada una
  con su metadata, configuración y registros) tiene que coincidir con la del
  plan; después se fija el sha256 de cada archivo del almacén y cada D5
  posterior (antes y después del VACUUM, y en `resume`) compara ese pin
  exacto (`payload_changed:chroma`). Un almacén cuyas colecciones se dan
  todas de baja se crea igual, vacío, para que los `schemas` de K sigan
  iguales a los de H (A1).
- **Registros sin contenido.** `plan` muestra `chroma` con conteos por
  almacén (`rebuilt`, `copied`, `dropped` y `other_files`: archivos dentro
  del almacén que no son de Chroma y que K no recibe, revisión SC4C-RV-5;
  el operador revisa antes de seguir si no es 0, SC4C-Q5) y los nombres de
  las bajas (`dropped_collections`); el evento `materialized` y la
  evidencia de preparación de K guardan conteos (almacenes, colecciones por
  clase, registros, excluidos), las bajas como `[almacén, colección]` y los
  pins (rutas y sha256). El snapshot
  conserva la exportación de las colecciones copiadas y de la de política
  (con los registros excluidos) hasta que se retira junto con H; de
  `home_assistant_routines` y de la heredada `user_memories` no exporta
  nada y el manifiesto sólo guarda el nombre (sus bytes sí están en la copia
  opaca de los archivos del SDK de H, ver (3b)).
- **Residual.** El texto de `home_assistant_routines` que ya está en H queda
  en H (y en los bytes opacos del snapshot) hasta que el retiro destruye H;
  la transición es lo que lo quita. Lo mismo vale para el residuo de heap de
  SC4C-Q3: un archivo HNSW de `data/chroma_db` de H puede tener texto de
  memoria (en la ronda 2 de la revisión el test lo mostró una vez en el
  server, con la `user_memories` heredada) y su copia opaca en el snapshot
  también, hasta
  el retiro o el `abort`; K no lo recibe porque cada almacén lo escribe un
  hijo limpio. SC-0 encontró un lector en producción
  (`routine_manager._handle_list` → `list_routines`, el intent de voz
  «listar mis rutinas»): después de la transición esa lista sale vacía hasta
  que se dicte una rutina nueva. El diseño deja a SC-R decidir entre la baja
  y una reconstrucción desde la tabla protegida `routines`; el usuario
  aceptó la baja (SC4C-Q1, 2026-11-16).
- **Hallazgo fuera de esta unidad.** El mismo comportamiento de Chroma vale
  para el servicio: los archivos HNSW que escribe `kza-voice` pueden guardar
  fragmentos de su heap (cualquier dato que el proceso tuvo en memoria), y
  un borrado no los limpia. Decisión del usuario (SC4C-Q3, 2026-11-16):
  después, como unidad propia con prioridad alta; la transición no lo
  empeora (K se escribe desde el hijo limpio y no copia los bytes de Chroma
  de H).
- **Límite de 600 s por almacén.** El usuario lo aceptó (SC4C-Q4,
  2026-11-16); SC-6b lo ajusta con la medición de tamaño de producción.

Tests: `tests/unit/tools/test_sc4c_chroma_closed_list.py`.

### Procedimiento del modo release (SC-5b) — runbook, sin release

El orden completo del operador, con cada salida, está en
[el runbook del modo release](../runbooks/2026-11-17-release-mode.md)
(diseño `sc-release-design-v4`, unidad SC-5b; evidencia
`data/autonomous-roadmap-20261102/t040-t041/sc5b-evidence.json`, local y
fuera de Git). Nada está instalado: el hueco de la sección siguiente sigue
abierto hasta SC-6b, y la primera transición tiene además las dos
condiciones de SC4B-Q1 y SC4B-Q2 (dentro de los 14 días de L4; `resurrected`
en 0 en el paso 0). Lo que el operador tiene que saber de antemano:

- **Punto de aceptación.** La release nueva queda aceptada con el retiro
  (`compact retire` terminado). Hasta ahí el tráfico de la casa no llega a
  K'; después, H ya no existe y un problema se arregla con otra transición
  (hotfix o rollback), nunca volviendo a H.
- **T0.** Sólo cuando el triple del NUEVO es estrictamente mayor que el
  instalado, como `$PY -I -B <tmp>/scripts/stage_runtime.pyc --root
  /home/kza/releases --offline install-controller <archivo NUEVO>` desde el
  archivo extraído (nunca el `$ST` instalado). Como la clausura del
  controlador incluye casi todo `src`, toda release que cambia `src`
  necesita un T0 con `CONTROLLER_REVISION` subido (revisión SC-R SCR-C-3).
  Después del T0, H sirve una vez con el controlador nuevo y la unidad se
  detiene y deshabilita otra vez antes del plan. No hay vuelta al
  controlador anterior (fila T0F). Un rollback no lleva T0.
- **Reparaciones que V no ejercita (SC-D6 A).** V saltea las reparaciones de
  una vez y los bucles de mantenimiento (lista medida por SC-0): un defecto
  ahí aparece recién después de aceptar, y arreglarlo no devuelve lo que
  borró.
- **`home_assistant_routines` se descarta en cada transición** (SC4C-Q1,
  SC4C-Q6): el arranque de K' la recrea vacía, así que después de cada
  transición «listar mis rutinas» por voz responde vacío hasta que se dicte
  una rutina nueva. La `user_memories` heredada de `data/memory_db` se
  descarta en la primera (SC4C-Q2b).
- **Colas de borrado de audio.** La premisa anterior («una cola de borrados
  pendiente deja rutas viejas de H») estaba equivocada: una cola no vacía
  bloqueaba toda compactación. Desde SC-1b las rutas de
  `pending_audio_deletions` y `audio_write_intents` se reubican en K y A1
  compara la proyección con esas rutas reubicadas (sección SC-1b).
- **Raíces con dueños sólo en ensayos.** Las raíces con dueños (archivos
  anteriores a SC-2) existen sólo en tests y ensayos de época 4 o menor, se
  recrean en cada uno y siguen en modo simple; producción no puede tener
  dueños (`runtime_admission_owners_present`,
  `runtime_admission_compaction_owners_present`).
- **Residuos aceptados.** Durante SS, HD y CI quedan tres copias (H, el
  snapshot y K') con los borrados por plazo en pausa, y durante T0F, PA o
  una parada del hueco, una; todo período con la unidad deshabilitada lleva
  la decisión escrita cada 48 horas. El borrado es sólo `unlink`. El
  snapshot conserva la copia opaca de Chroma de H hasta el retiro o el abort.
  Nada fuera de la raíz se alcanza.
- **Filas PA y DR, implementadas (SC-PA-DR).** Un head cerrado por un
  registro de recuperación limpio es fuente de un hotfix o un rollback; uno
  con regresiones confirmadas, sólo con `--accept-regressed-source` (fila
  PA). La copia en claro de `adopt --dry-run`, que corre siempre que no haya
  un log del modo release pendiente (después de la aceptación, y también en
  la salida T0F y en el rechazo `runtime_admission_not_closed` del T0,
  durante la primera transición), vive sólo bajo `<raíz>/admission/dry-run/`
  y la borra el próximo que toma la concesión (fila DR). Detalle en «Filas PA
  y DR (SC-PA-DR)». **Plazo de PA y DR (SC5B-Q1 y SC5B-Q2, respuesta del
  usuario del 2026-11-17):** las dos filas se implementan antes de DÍA 0 + 14
  × 24 h de R1; ya están en el código, y tienen que estar en el controlador
  instalado, revisadas (SC-R) y ensayadas (SC-6b) para una transición
  después del día 14.
- **Pendiente en el código, fallando cerrado.** La línea base salada de
  resurrección de K y el reencolado (el retiro sigue con escaneo absoluto;
  por eso el paso 0). Mientras tanto, una fuente regresada cuyas regresiones
  confirmadas incluyen una familia `resurrected_*` no llega al retiro
  (`runtime_admission_compaction_retire_resurrected`): vuelve a H con W.

Tests: `tests/unit/tools/test_sc5b_docs.py` (comandos y códigos del runbook
contra el código, el orden de los pasos y las salidas).

### Filas PA y DR (SC-PA-DR)

Unidad SC-PA-DR (2026-11-17; diseño `sc-release-design-v4`,
`mechanism.closed_witness` y `mechanism.dry_run_shadow`, filas PA y DR de
`crash_table`; respuesta SC5B-Q1/Q2 del usuario: antes del día 14 de R1;
evidencia `data/autonomous-roadmap-20261102/t040-t041/sc-pa-dr-evidence.json`,
local y fuera de Git). Nada instalado.

- **Testigo de cierre (`stage_runtime.closed_witness`).** Un head cerrado
  por recuperación (o el head de una vuelta a ese H) se prueba con el
  registro de recuperación inmutable que escribió `adopt_recovered_state`,
  aunque una prueba de cierre anterior pruebe el mismo binding (una
  regresión puede restaurar exactamente un estado cerrado antes); cualquier
  otro head, con su prueba de cierre como antes. El registro vale con su
  companion `recovery` verificado contra el registro, `new_binding_id`
  igual al binding del head y `checks.regressed` vacío (sin regresión del
  ledger ni de familias append-only, sin `resurrected_*`, sin
  `state_changed_after_close`, sin ledger faltante). Lo usan
  `verify_closed_binding` (plan P1/P2, A1, la activación de K en A2, el
  worker, el catálogo de copias), `_h_intact` y W0 (H después del start: el head de la intención,
  también a través del registro de vuelta de un abort a H), el snapshot
  (registra `source_witness`: `closure` con prueba y recibo, o `recovery`
  con el registro y su sha256) y la derivación del journal (exige que el
  testigo registrado sea el de la fuente). `compaction_verified`, el retiro
  de K' y las delegaciones siguen exigiendo una prueba de cierre.
- **Fuente regresada (R5-1).** Un registro con regresiones confirmadas no es
  testigo (`runtime_admission_recovery_regressed_source`) salvo para el H de
  un log planeado con `compact plan|start --accept-regressed-source <12 o
  más hex del sha256 del registro>`: sin la marca `plan` rechaza
  `runtime_admission_compaction_regressed_source_required` (nombra el
  sha256 y las familias); un prefijo corto, ajeno o sobre un head limpio
  rechaza `runtime_admission_compaction_regressed_source_mismatch`; la marca
  es sólo del modo release. El digest del plan y la intención ligan el
  sha256 completo (`intent.source_recovery_sha256`); el plan, A1, W0 y la
  activación de K en A2 lo pasan explícito (la rama de compactación de
  `activate`, que corre en el proceso del controlador cuando la autoridad
  de K difiere de la de H, p. ej. con rutas de cola de audio reubicadas, lo
  toma de la intención del log pendiente), y el worker lo liga desde la
  intención (`regressed_source_bound`) para el snapshot, el catálogo y la
  derivación.
- **Sombra del dry-run.** `adopt --dry-run` copia el estado sólo bajo la
  concesión, en `<raíz>/admission/dry-run/<id aleatorio>/` (0700, mismo
  sistema de archivos que el estado), nunca en `$TMPDIR`, y la borra al
  terminar. `AdmissionController.__enter__` borra todo lo que haya bajo
  `admission/dry-run/` justo después del `flock`, antes de cualquier otra
  comprobación y en toda compuerta (sin seguir enlaces;
  `runtime_admission_dry_run_shadow_unsafe` si `admission/` es un enlace).
  Residuo aceptado: entre un dry-run cortado y la próxima concesión hay una
  copia completa bajo la raíz.

Tests: `tests/unit/tools/test_sc_pa_dr.py` (rollback desde un head de
recuperación limpio; fuente regresada con plan, start, vuelta a H con W,
otra transición y retiro; la activación de K con filas de audio en cola,
cuya autoridad difiere de la de H; la transición siguiente sin la marca; la
derivación con un testigo cambiado; el dry-run cortado a mitad de la copia
con `SIGKILL`, sin nada en `$TMPDIR`, borrado por la concesión siguiente
aunque después rechace, también la de `install-controller` y la de una
compuerta de compactación; la marca fuera del modo release) y
`tests/unit/tools/test_sc5b_docs.py`.

### Sin transiciones gestionadas después de R1 hasta el modo release

Después de R1 no se puede activar ningún arreglo de código (release, rollback
o roll-forward gestionados) hasta que el modo release (SC-0, SC-1, SC-3a,
SC-3c, SC-4a/b/c, SC-R, SC-6b y SC-5b) esté desarrollado, revisado de forma
independiente y ensayado. Para SC-4b, el usuario respondió SC4B-Q1 y SC4B-Q2
el 2026-11-16 (ver «Salidas de V del modo release»): la primera transición
tiene que caer dentro de los 14 días de L4 y el paso 0 del runbook tiene que
dar `resurrected` en 0; si no se cumple alguna, esa parte se implementa
antes. Una release gestionada copiará entero el head,
verificará la copia con una corrida corta y retirará el estado anterior; nada
de eso está disponible hoy. Mientras tanto, el único rollback es L4 al contenedor de e9669ec
durante sus 14 días (con la frase 'revivir datos borrados'); a partir del día
14 (retiro Q5) tampoco queda ese rollback.

Cómo se acota ese hueco lo decidió el usuario el 2026-11-15: **SC-D5 = A**
(registro `u3b_sc_release_mode_questions` en
`t040-t041/user-decisions-20261110.json`, estado aceptado). Se acepta el
hueco: R1 sale apenas esté listo, el modo release se empieza enseguida y, ante
un problema grave de privacidad en el hueco, se apaga el servicio hasta
tenerlo. El retiro Q5 del día 14 no se posterga. Las opciones descartadas eran
B (postergar el retiro Q5 hasta tener el modo release revisado y ensayado,
conservando L4 a cambio de guardar más tiempo la copia heredada) y C
(postergar R1 hasta que el modo release esté casi listo). En la misma tanda
el usuario respondió A a SC-D2 (hasta 10 minutos sin servicio por cambio),
SC-D4 (las primeras versiones después de R1 no cambian la estructura), SC-D6
(corrida corta aislada antes de aceptar), SC-D7 (arreglo hacia adelante con el
servicio apagado, o L4 mientras exista; R1 nunca es destino del modo release)
y SC-D8 (si la prueba aislada registra un borrado o quita un permiso, servicio
apagado hasta revisar). SC-D1 y SC-D3 quedaron sin objeto en el diseño v4 del
modo release (propuesto, en revisión), que es donde esas respuestas se
aplican.

Hallazgo aparte, de 0a: un estado con unas 250 000 filas protegidas o más
produce un binding de más de 16 MB, y el controlador no puede leerlo
(`runtime_admission_unsafe_record`). Lo produjeron los 500 000 eventos con plazo
acortado del arnés de costo, al cerrar. Hay que revisarlo antes de una release
con datos reales.

Ver la evidencia [0a](../../data/autonomous-roadmap-20261102/t040-t041/u3b-0a-evidence.json)
y [0b](../../data/autonomous-roadmap-20261102/t040-t041/u3b-0b-evidence.json),
locales y fuera de Git.

Es bytecode para CPython 3.13, no un ejecutable nativo ni una barrera contra
descompilación. Requiere el intérprete y las dependencias del entorno de
ejecución. Ver [py_compile](https://docs.python.org/3.13/library/py_compile.html)
y [SourcelessFileLoader](https://docs.python.org/3.13/library/importlib.html#importlib.machinery.SourcelessFileLoader).

### Captura post-wake, wake en sombra y salud del STT (voice-capture) — desarrollado, sin release

Rama `voice-capture` (2026-11-17), desde el diagnóstico
`data/autonomous-roadmap-20261102/voice-diag-20261117/evidence.json` y las
respuestas del usuario (`voice_diag_questions_20261117`). Sin release: la
fecha la decide el usuario después de los tests del servidor y de la prueba en
vivo de la Cocina (`docs/runbooks/2026-11-16-cocina-acceptance.md`, «Prueba en
vivo de la captura post-wake»).

- V1, `rooms.endpointing.post_wake`: los tiempos se miden desde el wake (el
  pre-roll no cuenta); escucha >= 1 s y como mucho 5 s; cierra sólo tras voz
  después del wake y silencio sostenido: 300 ms por nivel (Silero no cuenta
  para el silencio porque su probabilidad se arrastra tras la voz), 600 ms si
  la voz sólo se confirmó con Silero, 150 ms si el parser ya tiene el comando
  completo (sólo con `early_dispatch`, hoy apagado); voz = nivel >= 1,8 × el piso de ruido medido de la
  habitación, o Silero del wake >= 0,5. Sin voz confirmada (>= 240 ms; la cola
  del «Nexa» o un golpe aislado no alcanzan) en 2,5 s descarta sin STT
  (`[endpoint] code=drop_no_speech`, cuenta como captura vacía para el
  AmbientGuard). Al tope de 5 s con voz la captura queda incompleta y el router
  la rechaza sin STT (`gate_rejected:incomplete_capture`, como el timeout
  anterior) mientras el usuario no responda VC-Q2; `dispatch_at_max_utterance:
  true` la despacha y entonces exige la wake word en la transcripción
  (`gate_rejected:max_utterance_without_wake`). El follow-up arranca sólo en la
  habitación que abrió la conversación (wake, comando aceptado o barge-in) y
  con >= 240 ms seguidos de voz por nivel; Silero solo no lo abre (no distingue
  TV de personas) y un golpe aislado tampoco. El `command_duration`
  que estaba fijo en `main.py` pasa a `rooms.endpointing.command_duration_s`
  (endpoint legacy y AudioManager). `enabled: false` restaura el endpoint
  anterior.
- V2, `rooms.wake_word.verification.mode: shadow`: para wakes acústicos, la
  transcripción debe contener una variante de `rooms.wake_word.words` (regla
  `missing_wake` del gate). En sombra sólo registra `would_reject`; `enforce`
  rechaza como `gate_rejected:missing_wake`. Contadores para la revisión de 7
  días en `data/wake_verification.json` del estado (archivo nuevo, `0600`,
  sólo fecha, habitación, código, score y VAD; nunca texto; contadores por día
  y eventos se borran a los `keep_days` = 14 días). Cada wake loguea
  su VAD.
- V3, `stt.primary_health`: cada fallo del STT primario se loguea con un código
  saneado (`Tipo:categoría:hash`); tras 3 seguidos, una alerta por episodio en
  el AlertManager (MEDIUM, texto en español), recarga del Whisper (libera y
  después carga, bajo el lock del modelo, como mucho una cada 600 s) y,
  mientras falle, el transcript del shadow (Parakeet, CPU)
  reemplaza al primario (`stt_source=shadow_fallback`). Se recupera con el
  primer éxito (alerta LOW). Límite de la recarga: no carga mientras el modelo
  anterior siga referenciado; si `unload_model` falla, no carga nada y conserva
  el modelo anterior (cargar ahí podría dejar dos copias en cuda:1). Que la
  VRAM se libere de verdad depende de CTranslate2 y no se verificó en la GPU.
  Si la carga falla, el modelo queda ausente y las transcripciones fallan al
  instante sin intentar cargarlo; sólo la siguiente recarga permitida (600 s)
  vuelve a cargar. Mientras recarga o está degradado, los comandos van directo
  al respaldo (un comando cada 30 s, y el primero tras una recarga exitosa,
  prueba el primario). Fallas intermitentes (>= 4 en los últimos 12 intentos,
  sin 3 seguidas) dan una sola alerta MEDIUM. El panel de salud muestra el
  subsistema `voice_pipeline` (estado del primario, contadores y códigos
  saneados, y los conteos de `[endpoint]`), degradado durante un episodio. La
  transcripción del respaldo queda marcada como `stt_source=shadow_fallback`
  en el resultado y en los eventos de calidad ASR (`stt_fallback`), sin el
  linaje del primario.

Sin dependencias ni modelos nuevos; la asignación de GPUs no cambia.

### Índice de comandos: puntero, colección legacy y rollback (U3b-0d2)

Desde U3b-0d2, la sincronización de comandos del runtime construye el índice
en una colección nueva (`home_assistant_commands__<fecha><hex>`) y la activa con
una sola escritura en el puntero: la metadata `active` de la colección
`home_assistant_commands__pointer`. Hasta que no está completa, las búsquedas
siguen usando el índice anterior. Si falla, o se cierra a mitad, la colección
nueva se descarta y queda el índice anterior.

- **Rollback a código anterior a 0d2.** Ese código abre sólo
  `home_assistant_commands` e ignora el puntero. Por eso ninguna
  sincronización borra nunca esa colección legacy: tras un rollback se sirve
  tal como estaba antes de la primera sincronización 0d2 (o tras la última
  corrida de `scripts/sync_ha_to_chroma.py` anterior a 0d2). Si para entonces
  quedó vieja, después del rollback corré `scripts/sync_ha_to_chroma.py` con el
  código restaurado, que escribe esa misma colección. La colección legacy se
  puede retirar recién en la release que deje de admitir rollback a antes de
  0d2.
- **Herramientas offline.** `scripts/sync_ha_to_chroma.py` (la vía real de
  sincronización hoy), `tools/clean_contaminated_chroma_docs.py` y
  `scripts/test_voice_command.py` resuelven la colección activa con
  `src/vectordb/command_index.py`, el mismo resolvedor que usa el runtime.
  `--wipe` borra y recrea la colección activa con el mismo nombre, así que el
  puntero sigue nombrándola.
- **Puntero a una colección que falta.** Nunca se crea vacía en silencio: el
  runtime registra el error, sirve la colección legacy y `/api/health` muestra
  `command_index` como `degraded` hasta la próxima sincronización correcta.
- **Escritura del puntero con resultado incierto.** No se sabe si el puntero
  quedó en la colección vieja o en la nueva. Por eso se conservan las dos en
  disco, y el proceso rechaza más sincronizaciones hasta reiniciarse
  (`chroma_sync_restart_required`; `command_index` aparece como `degraded`). Al
  reiniciar se abre la que nombre el puntero.
- **Restos en disco aceptados.** Si el proceso muere entre la escritura del
  puntero y el borrado de la colección anterior, esa colección anterior queda
  en disco. Una colección `__<fecha>` sobrante la limpia la próxima
  sincronización. La legacy se conserva siempre, por el rollback.

### Lanzador gestionado (U3b-0d) — desarrollado, no instalado

Estado: código y ensayos en la rama de desarrollo. **No está instalado** y no
se instala hasta la ventana de R1. Antes hacen falta U3b-0e (el primer head
gestionado a partir del estado de producción, con rutas absolutas), correr 0d1
en Linux y medir en el server los presupuestos de parada (L6) y del watchdog
(L2).

Huecos que quedan en Linux (la revisión independiente de U3b-0d, P12 y P13):

- **T-C8** (cgroup real) y **T-C10** (matriz de parada en el server) no se
  ensayaron: el supervisor falso de los tests usa un grupo de procesos, no un
  cgroup, y no aplica `NotifyAccess=main` en macOS. Correrlos en el server o
  en CI Linux antes de instalar. Agregar ahí dos casos que el falso no
  modela: la parada pedida por `STOPPING=1` (systemd arma `TimeoutStopSec`) y
  `EXTEND_TIMEOUT_USEC` durante la parada. T-C8 verifica además el parseo de
  `0::/user.slice/.../kza-voice.service` que usa la limpieza de rezagados.
- **Drop-in sudo en la misma ventana que R1.** Sin él, cada reinicio del host
  deja el head `unknown` (el manager tiene 5 s) y la recuperación por
  `boot_id` lo adopta sin límite de 24 h (L8): la seguridad queda entera en
  los chequeos de R-b. Aplicarlo en la misma ventana autorizada que la
  instalación (ver "Presupuesto de reinicio").

Cómo arranca:

- systemd (`kza --user`) inicia el controlador:
  `/home/kza/kza/.venv/bin/python -I -B /home/kza/releases/controller/current/scripts/stage_runtime.pyc --root /home/kza/releases --offline serve --stop-grace <S> --recover bounded --unit-timeout-stop <T>`.
- El controlador es el proceso principal y el único que escribe el head.
  Lanza `src.main` del slot del head con `cwd` en su `runtime`,
  `PYTHONPATH` en ese runtime, `CONFIG_PATH=config/settings.yaml` y el ticket.
  Hereda el lease y dos pipes privados: notificaciones y control.
- **Umask del hijo (revisión F5, 2026-11-14).** El controlador lanza el hijo
  con `umask=0o077` (`stage_runtime.MANAGED_CHILD_UMASK`); lo mismo hace
  `AdmissionController.run` para corridas de operador y ensayos. Sin eso el
  hijo heredaba el `0002` de `kza` y lo que `src/` crea en
  `slots/<slot>/state` quedaba abierto: directorios intermedios `0775`,
  archivos `0664`, bases SQLite y sus `-wal`/`-shm` `0644` (medido en Mac y
  Linux). Ahora es `0700`/`0600`. El trabajador de compactación
  (`compaction.run_worker`, que `compact start` lanza desde un shell de `kza`)
  usa la misma umask (revisión S1). Las sondas de lectura (candidato de
  `activate` y `path_preflight`) no crean entradas propias; un `-shm` que
  SQLite cree al abrir una base `?mode=ro` copia el modo de esa base, no la
  umask. El drop-in
  agrega `UMask=0077` como defensa adicional; también cubre al controlador y a
  un `console.log` que systemd cree de cero (queda `0600`, medido con
  systemd 255). Un archivo que otro lector necesite conserva su `fchmod`
  explícito: `audio_health.json` sigue `0644`, aunque vive bajo `data/` `0700`.
  Consumidores revisados antes del cambio: ninguna unit, timer o crontab de
  `kza` ni de `yo` lee el estado del slot. El único acceso de otra cuenta es la
  ACL de `yo` sobre `state/logs/console.log` (lectura, con tránsito `--x` por
  los directorios), y ese archivo lo abre systemd, no el hijo. `adoption.py`
  ya usaba `umask=0o077` para su hijo de esquema.
- Unit: `Type=notify`, `NotifyAccess=main`, `KillMode=mixed`,
  `RestartPreventExitStatus=69 78`, `Restart=on-failure`,
  `StartLimitBurst=5` en 300 s. Es un desvío registrado en
  `SERVER_CONVENTIONS.md` (excepción R10 #6).
- **Límite de arranques** (U3b-0e4, medido con systemd real): cinco
  arranques en 300 s agotan el límite, contando los reinicios automáticos.
  Un crash seguido de cuatro arranques rechazados con 69 alcanza. El
  siguiente arranque falla con «Start request repeated too quickly» sin
  ejecutar nada. Después de resolver la causa, correr
  `systemctl --user reset-failed kza-voice.service` y recién entonces
  arrancar.
- El hijo nunca recibe `NOTIFY_SOCKET` ni `WATCHDOG_*`. Escribe líneas en su
  pipe y el controlador las valida y las retransmite: READY una sola vez,
  WATCHDOG sólo después de READY, STATUS, STOPPING y EXTEND_TIMEOUT_USEC.
  El hijo avisa `ARMED=1` cuando su manejador de SIGTERM existe. El
  controlador consume esa línea, no la retransmite y retiene un stop hasta
  recibirla (máximo 10 s), para que un SIGTERM no llegue al intérprete antes
  de ese manejador.
- SIGTERM al controlador sólo escribe un byte en un self-pipe. No se enmascara
  ninguna señal. El controlador reenvía SIGTERM al hijo y recién después
  consulta `systemctl is-system-running`. Si el sistema está apagándose,
  escribe `fast` en el pipe de control y acota el plazo con el presupuesto del
  manager.
- El cierre se sella con una sola regla: recibo exacto, hijo cosechado,
  registro `receipt-states/` cuando el slot lo declara (`release.json`), y el
  manifiesto del estado recalculado bajo el lock antes de abrir cualquier
  base. El código de salida del hijo no decide.

Códigos de salida del controlador:

| Código | Significado | systemd |
|---|---|---|
| 0 | parada limpia, head cerrado | no reinicia |
| 75 | el hijo terminó solo y se selló | reinicia |
| 68 | head `unknown` o `running`, apto para una recuperación automática | reinicia |
| 69 | head `unknown` o `running`, no apto | no reinicia |
| 78 | rechazado antes de CAS `running` (selección, contrato, época, rutas, lease ocupado) | no reinicia |
| 70 | error transitorio antes de CAS `running` | reinicia dentro de StartLimit |

Registros nuevos en `admission/`: `runs/<run>.json` y `runs/<run>.exit.json`
(inmutables), `recoveries/`, `seal-intents/`, `controller.json` y
`launcher/` (instalaciones del controlador y del lanzador, retiro). Todos son
sin contenido.

Recuperación al arrancar (L1 = A):

- **R-a** (con recibo): si el controlador murió después del recibo, el
  arranque siguiente completa el sello. No gasta cupo. Si el manifiesto o el
  digest de estado cerrado no coinciden, el estado cambió después del cierre:
  no se sella y sólo el operador puede adoptar.
- **R-b** (sin recibo): se adopta el estado una vez cada 24 h. Nunca dos veces
  seguidas sin un cierre limpio en el medio. Antes se comprueba:
  - que no haya escritores;
  - `quick_check` de cada SQLite, con recuperación del WAL;
  - que los JSON parseen;
  - la estructura de los segmentos de Chroma (sin abrir el SDK);
  - que el registro de borrados y las familias de sólo-agregar avancen;
  - que ningún id borrado haya resucitado en los almacenes de payload. En
    Chroma, los hechos se ven como `embeddings.embedding_id` (así guarda
    `src/memory/long_term.py` cada hecho: `ids=[fact.id]`), por colección,
    más la cola `embeddings_queue` aún no aplicada. Un hecho con su borrado
    todavía pendiente en `index_jobs` o `policy_index_deletions` no cuenta;
  - la propiedad de destinos;
  - la equivalencia de componentes manuales.
- Antes de recuperar y entre cada paso de R-b (y antes de adoptar) el
  controlador envía `EXTEND_TIMEOUT_USEC` de 120 s, para que
  `TimeoutStartSec` no caiga a mitad de una adopción.
- **EXTEND no alcanza solo** (U3b-0e4, F1, decisión del usuario C): con
  systemd 255, un `systemctl --user daemon-reload` en el manager de `kza`
  descarta toda extensión y vuelve a armar el plazo desde `TimeoutStartSec`.
  Un arranque que ya lo superó se corta en el acto, pero el corte es seguro:
  - durante la recuperación o la activación, el controlador sale antes del
    CAS `running`: el head no cambia y R-b se repite en el próximo arranque;
  - durante el arranque del hijo (después del CAS, antes de READY), el
    controlador reenvía SIGTERM, el hijo deja su recibo y el head queda
    `closed` (el ensayo sin READY lo muestra).

  En los dos casos el arranque falla. Por eso:
  - `--timeout-start` vale 240 s por defecto (`DEFAULT_TIMEOUT_START`).
    Cubre con margen el arranque del lado del controlador medido en el
    server sobre un estado adoptado de 161 MB (500k eventos): 12,2 s el
    primer arranque y 20,1 s un crash con R-b hasta READY, con un hijo de
    ensayo. La activación crece lineal con las filas. La carga de modelos de
    la aplicación real (STT, TTS, speaker en cuda:1, BGE-M3, Chroma, HA) no
    está en esa medida: se mide en W8/W9 del runbook y, si el arranque hasta
    READY pasa de 120 s (la mitad del plazo), se sube `--timeout-start`
    antes de depender de él. `src/` no envía `EXTEND_TIMEOUT_USEC`: el
    plazo cubre sonda, activación, R-b y carga de modelos.
  - `install-launcher` acepta solo segundos enteros entre 30 y 3600 y, tras
    instalar, relee `TimeoutStartUSec` y `TimeoutStopUSec` con
    `systemctl show`. Si no coinciden con `managed.conf` (un valor que
    systemd rechazó, o un drop-in posterior como el de `systemctl --user
    edit`), sale con 78 (`runtime_admission_launcher_verification`).
  - Regla de operación: mientras `kza-voice.service` esté `activating`
    (`systemctl --user show -p ActiveState kza-voice.service`), no correr en
    el manager de `kza` ni `systemctl --user daemon-reload` ni los comandos
    que recargan implícitamente: `enable`, `disable`, `reenable`, `mask`,
    `unmask`, `link`, `preset`, `revert` y `edit`. Esperar a `active`,
    `failed` o `inactive`; un `start` bloqueante vuelve recién con READY o
    con el fallo, pero un reinicio automático en curso también es
    `activating`. `install-launcher`, `uninstall-launcher` y los comandos de
    adopción ya rechazan con la unidad activa o arrancando.
  - Recargas externas: una actualización del paquete systemd (por ejemplo,
    unattended-upgrades) re-ejecuta los managers de usuario. No está
    verificado que el re-exec descarte la extensión igual que un reload,
    pero conviene suponerlo: no actualizar systemd durante una ventana ni
    con kza-voice arrancando.
- **Recuperación por arranque**: si el run quedó del boot anterior (reinicio o
  corte de luz), hay una recuperación por `boot_id` y no gasta el cupo de 24 h.
- **Operador** (`UNKNOWN` sin adopción automática): primero
  `stage_runtime --root /home/kza/releases --offline adopt --dry-run`, que
  imprime el sha del head, si R-a sigue siendo posible (recibo, registro y
  manifiesto) y sólo conteos. Es de sólo lectura: no abre nada del estado
  para escribir y corre los chequeos de bases sobre una copia privada (con su
  `-wal`, sin `-shm`), así que no puede romper un R-a pendiente. La copia
  está en `<raíz>/admission/dry-run/<id>/` (0700, nunca en `$TMPDIR`) y la
  borra el próximo que toma la concesión si el dry-run se corta (fila DR,
  «Filas PA y DR (SC-PA-DR)»). `adopt --confirm` imprime además
  `recovery_record_sha256`, el que pide `--accept-regressed-source`. Después
  `adopt --confirm <sha> --confirm-regressions <familias>`, con las familias
  exactamente como las informó el dry-run.
- Un head recuperado no puede delegar hasta un cierre limpio posterior (desde SC-2
  ningún head puede delegar).

Instalación en la ventana de R1 (bajo `kza`, sin sudo; el servicio detenido):

1. Bajo `yo`: construir la release con `scripts/build_runtime.py`. Su
   `release.json` declara `launcher_contract`, `format_epoch`,
   `receipt_state` y `controller_epoch`, tomados del propio
   `scripts/stage_runtime.py` de esa revisión.
2. Transferir el archivo y `systemd/kza-voice.service.d/managed.conf.template`.
3. Extraer el archivo en un directorio temporal y correr
   `$PY -I -B <tmp>/scripts/stage_runtime.pyc --root /home/kza/releases --offline install-controller <archivo>`,
   con `$PY` = `/home/kza/kza/.venv/bin/python` (un `.pyc` no es ejecutable
   solo; corrección SC-5b, R4-OPS-6b).
   Instala en `controller/<commit>` y cambia `controller/current` de forma
   atómica. Antes del cambio hace fsync de cada archivo y directorio del
   árbol, y después fsync del directorio `controller/`. Cada `serve` vuelve
   a verificar los hashes de `controller/current` contra su `release.json` y
   rechaza con 78 (`runtime_admission_controller_tampered`) una instalación
   dañada. Rechaza bajar de época e informa qué slots quedan fuera de
   rollback.
4. Con `systemctl --user stop kza-voice.service` ya hecho, correr
   `controller/current/scripts/stage_runtime.pyc ... install-launcher`.
   - Argumentos: `--unit-dir ~/.config/systemd/user/kza-voice.service.d`,
     `--template`, `--python /home/kza/kza/.venv/bin/python`,
     `--stop-grace`, `--timeout-start` (por defecto 240 s, ver arriba),
     `--timeout-stop`, `--watchdog-sec`,
     `--console-log /home/kza/releases/logs/console.log` y, si no es el de
     siempre, `--logrotate-conf` (por defecto
     `/home/kza/releases/console-logrotate.conf`).
   - Rechaza si la unidad está activa o si no existe un head gestionado.
     También rechaza un `--console-log` bajo `slots/`.
   - Antes de tocar el directorio de la unidad corre los mismos chequeos que
     `serve` al arrancar, sobre el binding del head: selección, contrato,
     época, compañeros y el preflight de rutas con el intérprete de
     `--python`. Con el `settings.yaml` del repo (rutas relativas de modelos
     y log, `training.output_dir` fuera del estado) rechaza: no instala algo
     que saldría con 78. Reubicar esas rutas es trabajo de U3b-0e; esto sólo
     detecta y rechaza. También exige que el directorio de `--console-log`
     exista y sea escribible (`StandardOutput=append:` no crea
     directorios) y que la configuración de logrotate rote esa ruta con
     `copytruncate`. Rechaza con `runtime_admission_launcher_preflight` y la
     lista exacta de problemas.
   - Rotación de la consola nueva: `stage_runtime ... console-logrotate
     --console-log /home/kza/releases/logs/console.log` imprime la estrofa
     (horaria, 24 rotaciones, `maxage 1`, compresión, `copytruncate`). `kza`
     la agrega a `console-logrotate.conf` en la misma ventana; el timer
     existente la ejecuta. No requiere sudo.
   - Guarda las propiedades previas: `Environment` sólo como digest.
   - Mueve `release.conf` a `release.conf.unmanaged`, escribe `managed.conf`,
     hace `daemon-reload` y verifica `Type`, `NotifyAccess`, `KillMode`,
     `RestartPreventExitStatus` (exactamente `69 78`), `ExecStart` y
     `WorkingDirectory` contra lo generado, e informa `DropInPaths`.
     Cualquier diferencia sale con 78 y deja `managed.conf` en su lugar para
     que el operador la vea.
   - Imprime el texto del drop-in sudo.
   - La herramienta de publicación del 2026-10-29 queda retirada: su
     `--apply` y su `--rollback` se niegan (`managed_launcher_installed`) si
     existe `managed.conf`, porque `release.conf` ordena después y pisaría
     el `ExecStart` del controlador.
5. Arrancar sólo con un head gestionado: sin head, `serve` sale con 78.

Valores (L2, L6): `stop_grace` es el máximo entre 3 veces la mediana medida
en el server, 1,5 veces el máximo observado y 120 s. `TimeoutStopSec` es
`stop_grace` más el sello medido más 30 s. El sello toma tres snapshots
completos: del hijo, el previo al sello y el cerrado. Cada uno cuesta unos
2,6 s por millón de filas del registro de borrados (0d3 K7). `WatchdogSec`
queda en 0 hasta medir el bucle (0d2) y después vale
`max(60 s, 3 × el peor bloqueo)`.

Presupuesto de reinicio (decisión N1 = B del usuario):

- Hoy el manager de `kza` tiene 5 s para cerrar (el `timeout.conf` del paquete
  systemd).
- El controlador lee el presupuesto efectivo en cada arranque:
  `systemctl show user@<uid>.service -p TimeoutStopUSec`.
- Si no alcanza (presupuesto < `TimeoutStopSec` de kza-voice + 10 s), opera
  en modo de respaldo:
  - publica `STATUS=reboot budget insufficient (<n>s)`;
  - en un apagado pide cierre rápido y acota el plazo a presupuesto − 2 s;
  - si no llega a cerrar, el arranque siguiente usa la recuperación por
    `boot_id`.
- La excepción la aplica sólo el usuario u operador, en una ventana
  autorizada. Esta unidad no la crea.

  ```
  sudo install -d -m 0755 /etc/systemd/system/user@1000.service.d
  printf '[Service]\nTimeoutStopSec=<B>s\n' | sudo install -m 0644 /dev/stdin /etc/systemd/system/user@1000.service.d/zz-kza-stop-timeout.conf
  sudo systemctl daemon-reload        # no reiniciar user@1000.service
  ```

- Verificación de sólo lectura:
  - `systemctl show user@1000.service -p TimeoutStopUSec,DropInPaths`
    (B, y el drop-in `zz-` último);
  - `systemctl cat user@1000.service`;
  - `ls /etc/systemd/system.control /run/systemd/system.control`.
- Repetir la verificación después de cada actualización del paquete systemd.
- Registrar el valor y la fecha aquí y en `SERVER_CONVENTIONS.md` (#7).
- Para reinicios planificados, igual conviene detener kza-voice antes.

Dependencias que quedan fuera de los slots: el intérprete y las dependencias
de `/home/kza/kza/.venv` (controlador e hijo) y el directorio de modelos.
El retiro Q5 del estado de b0866e8 los excluye y rechaza si su destino los
contiene o está dentro de ellos (ver «Retiro del estado congelado de b0866e8»).

Volver al contenedor de e9669ec (L4 = A):

```
stage_runtime --root /home/kza/releases --offline uninstall-launcher \
  --unit-dir ~/.config/systemd/user/kza-voice.service.d \
  --user-decision <id> --phrase 'revivir datos borrados' --legacy-slot <slot>
```

- Requiere la unidad detenida y verifica el commit y el manifiesto del slot
  legacy. Sólo vale el contenedor de e9669ec (`LEGACY_COMMITS` =
  `(LEGACY_CONTAINER_COMMIT,)` en `scripts/managed_launcher.py`; decisión
  `r1e_questions_20261118`, R1E-Q2/Q4/Q5). Cualquier otro commit rechaza
  `runtime_admission_uninstall_legacy_commit`: el runtime propio de b0866e8,
  el contenedor anterior de 63ea8d3 y el slot de 2c5adb4 incluidos. Los dos
  aborts usan la misma lista.
- Comprobación de enlaces (U3b-0e5): cada enlace directo del `runtime` y del
  `state` del slot legacy (en e9669ec, los cinco del contenedor) tiene que
  resolver, salto por salto, a algo que existe, y `config` y `data` a
  directorios. Un enlace colgante, en bucle o faltante rechaza con
  `runtime_admission_uninstall_legacy_state` antes de tocar el directorio de
  la unidad. `adoption-abort` (W-abort-1 y W-abort-2) hace la misma
  comprobación antes de cambiar nada.
- Un slot que nombra un registro o una intención de retiro Q5 rechaza con
  `runtime_admission_uninstall_legacy_retired` y el nombre de ese archivo
  en `problems` (`legacy-retirement/retired-<plan>.json`).
- Mueve `managed.conf` a `managed.conf.disabled` y genera `legacy.conf`:
  `Type=simple`, sin watchdog, y `PYTHONPATH` y `WorkingDirectory` explícitos
  del slot legacy. Nunca usa `current`. Parte de `release.conf.unmanaged`:
  de cada línea `Environment=` quita sólo la asignación `PYTHONPATH` (las
  demás variables de la misma línea, como `CONFIG_PATH`, quedan) y copia
  tal cual las directivas fuera de la lista fija (`LimitNOFILE`,
  `UnsetEnvironment`, `KillMode`...). El resultado y el registro listan las
  directivas copiadas.
- Verifica con `systemctl --user show`.
- El único cambio en `admission/` es `launcher/uninstall-<ts>.json`.
- El código viejo corre sobre los datos anteriores a R1: devuelve recuerdos,
  conversaciones y perfiles de voz borrados desde entonces. Decide el usuario,
  no el operador.

Rollback gestionado (**no disponible desde SC-2 hasta el modo release**: bajo
un head gestionado, `activate` de otro slot rechaza con
`runtime_admission_transition_requires_release_mode`; ver «Sin transiciones
gestionadas después de R1 hasta el modo release». Lo que sigue describe los
chequeos que el modo release reutilizará y los que ya aplica `serve`): un slot
destino necesita `launcher_contract >= 3` y
`format_epoch` igual o mayor que el mínimo del controlador. Si no, `serve`
lo rechaza con 78 (`runtime_admission_launcher_contract_missing` o
`runtime_admission_rollback_format_closed`). Con un controlador instalado
(`admission/controller.json`), `stage_runtime activate` aplica los mismos
enteros y rechaza antes de mover el head o `current`, así un activate del
operador no deja un arranque que sale con 78. La API acepta
`launcher_checks=False` sólo para ensayar ese rechazo de `serve`; la CLI no
lo expone. Desde SC-2, bajo un head gestionado `run()` sólo corre el slot
del propio head; correr linajes viejos queda para raíces de prueba con
`LEGACY_MANAGED_TRANSITIONS`.

Evidencia: `data/autonomous-roadmap-20261102/t040-t041/u3b-0d-evidence.json`
y, para las correcciones de la revisión independiente,
`u3b-0d-fix-evidence.json` (local, fuera de Git).

### Retiro del estado congelado de b0866e8 (Q5, P7) — desarrollado, no ejecutado

Estado: `retire-legacy-state` y la comprobación de enlaces están en la rama de
desarrollo (U3b-0e5). No se ejecutaron en producción: el estado de b0866e8
sigue siendo el estado servido hasta la ventana de R1. La ventana de R1 no se
programa hasta que los dos estén revisados y ensayados (T-A17; ver el
[runbook de la ventana](../runbooks/2026-11-13-r1-adoption-window.md)). Al
2026-11-15 esa compuerta está cumplida: revisión independiente aprobada en su
ronda 2 (2026-11-13, `t040-t041/u3b-0e5-independent-review.json`) y T-A17
compilado en Linux, que pasó en el server en SC-6a y en la suite batch de R1-B
sobre el árbol de c6462e5. La compuerta no es lo único pendiente antes de la
ventana. La re-revisión de la ronda de arreglos de SC-6a (cbb509c, 9ec7d95)
y la revisión independiente de la ronda 1 de arreglos de R1-B1/R1-B2 se
hicieron el 2026-11-15 (RR-1 a RR-4, RR2-1 y RR2-2); sus arreglos (19dd44e,
b02116c, 357dcb5) cambiaron código, y R1 se re-fijó desde 357dcb5. La
re-revisión independiente de esos arreglos (ronda 2, 2026-11-16) aprobó; sus
hallazgos bajos RR3-1 y RR3-2 se corrigieron sin tocar código de ejecución,
así que R1 sigue fijado desde 357dcb5. Las revisiones del lado de desarrollo
previas a la ventana están hechas. R1E (2026-11-18) cambió código de
ejecución y R1 se re-fijó desde d045156; R1E-2 (2026-11-18, decisiones
`r1e_questions_20261118`) volvió a cambiarlo (alcance del día 14 y L4 sólo
hacia e9669ec) y R1 se re-fijó desde 7752570. La ventana espera la revisión
independiente de esa ronda y la suite batch. La suite batch completa sobre 51fbc43 pasó en la Mac y en el server (`u3b/full-51fbc43/RESULT.md`), y el visto bueno visual del panel de R1-B2 sigue con el usuario.
Después quedan los pasos del operador
(transferir y verificar el R1 fijado, el drop-in sudo de W7, el arranque real
hasta READY de W8); la ventana la programa el usuario, como operador, y sigue
sin programarse.

Plazo (decisiones del usuario P7 = B y `u3b_0e5_retirement` Q1 = A, ya
registradas):

- DÍA 0 es el primer cierre limpio gestionado: el sello de cierre de W9 (el
  `ended_at` del registro de salida de esa corrida).
- «Día 14» quiere decir DÍA 0 + 14 × 24 h, no un día de calendario. Hasta ese
  momento vale L4 hacia el contenedor de e9669ec, con decisión registrada y la frase.
- Desde DÍA 0 + 14 × 24 h el operador corre el retiro. El comando rechaza
  `runtime_admission_legacy_retire_too_early` un segundo antes y
  `runtime_admission_legacy_retire_no_clean_close` si todavía no hubo un cierre
  limpio gestionado.
- Responsable: el usuario, como operador de `kza`. En W10 se anota la fecha
  exacta en el handoff y aquí.

Qué borra (Q2 = A y R1E-2, decisiones `r1e_questions_20261118`), en este
orden, siempre desde rutas fijas y nunca desde la línea de comandos
(`legacy_targets` en `scripts/adoption_commands.py`):

- `slots/release_20261028T145442Z_b0866e8/state`, tomado del registro de
  adopción;
- `activation-report.json`, `live-report.json` y `prepare-report.json`, junto
  a ese estado en el slot de b0866e8 (R1E-Q3). Se revisaron sólo leyendo el
  2026-11-18 (nombres de claves y clases de valor: ids de release, commits,
  rutas bajo `/home/kza`, estados de unidades, conteos y la URL del panel en
  la LAN; sin datos personales). Están eximidos por nombre exacto y sólo como
  archivos regulares; cualquier otra entrada junto a `runtime/` y `state/`
  sigue rechazando `runtime_admission_legacy_retire_slot_unexpected`;
- las diez bases SQLite que la publicación de e9669ec (2026-10-29 15:50) copió
  de ese estado: `backups/release_20261029T185008966364Z_e9669ec/databases/`
  (el nombre del respaldo es el del contenedor del registro de adopción; ver
  `backup_databases` en `tools/publish_voice_order.py`, que
  `tools/publish_voice_loading.py` reutiliza), y las diez de la publicación de
  las 14:48, `backups/release_20261029T174830709997Z_63ea8d3/databases/`
  (R1E-Q1). Si el segundo conjunto ya no está, el plan lo anota como ausente.
  Cada conjunto tiene que ser exactamente el revisado:
  - diez archivos regulares `.db` o `.sqlite3`, con cabecera SQLite;
  - cada uno con el mismo nombre relativo que una base del `data/` del estado
    congelado;
  - ningún otro archivo (`-journal`, `-wal` incluidos), ni enlaces, ni
    montajes.

  Si no, rechaza `runtime_admission_legacy_retire_backups_set`. La corrida de
  prueba lista sus nombres relativos para revisarlos;
- los slots superados enteros (R1E-Q2/Q4/Q5): `release_20261029T174830709997Z_63ea8d3`
  (su runtime y los cinco enlaces de su `state/`, cuatro al estado de b0866e8)
  y `release_20261028T144002Z_2c5adb4` (un runtime y una copia más vieja del
  estado). Cada uno se reconoce por su nombre exacto y el commit completo de
  `runtime/release.json` (`63ea8d319b50506cc37ba73df1e41c87a41054cf` y
  `2c5adb4f3825e94bff86ef5ccff4e1391a1bc012`); tiene que tener sólo
  `runtime/` y `state/`, ningún archivo especial, y el de 63ea8d3 sólo enlaces
  en `state/`. Otra forma rechaza `runtime_admission_legacy_retire_superseded`
  con el slot en `problems`; uno que ya no está se anota como ausente.

Qué queda:

- el contenedor de e9669ec (sólo enlaces, que quedan colgando) y el runtime de
  b0866e8;
- `release.conf`, `config` y `.env` de los dos respaldos del 2026-10-29;
- los demás respaldos de `backups/`;
- `/home/kza/kza`, con sus modelos y su `.venv`.

Cada uno es una exclusión comprobada. Si un destino contiene alguno, o está
dentro de alguno (también a través de un enlace), rechaza
`runtime_admission_legacy_retire_excluded:<cuál>`: `models_root`, `venv`,
`legacy_checkout`, `backups_kept`, `r1_slot`, `head_slot`, `controller`,
`current` o `head_state`.

Los slots superados y los informes los tiene T-A17 desde W0 (revisión
R1E-RV2): el día 14 compilado los retira en un solo paso y después L4 rechaza
los cuatro slots (R1E-2).

Cómo se corre (bajo `kza`, nunca como root), en una sesión nueva y con el
controlador instalado, como W4 (el `$ST` extraído de la ventana ya no existe;
re-revisión RR-1):

```bash
PY=/home/kza/kza/.venv/bin/python
ST="$PY -I -B /home/kza/releases/controller/current/scripts/stage_runtime.pyc --root /home/kza/releases --offline"
$ST retire-legacy-state --decision <id P7> --container-slot release_20261029T185008966364Z_e9669ec --dry-run
systemctl --user stop kza-voice.service
$ST retire-legacy-state --decision <id P7> --container-slot release_20261029T185008966364Z_e9669ec --confirm <plan_sha256>
systemctl --user start kza-voice.service
```

- La corrida de prueba sólo lee. Imprime:
  - `plan_sha256`, `eligible` y `eligible_from`;
  - conteos del estado (archivos, directorios, enlaces, bytes y los nombres de
    primer nivel);
  - los nombres relativos de las diez bases de cada respaldo (`backups` y
    `superseded_backups`), los informes presentes (`slot_reports`) y los dos
    slots superados con sus conteos y su commit (`superseded_slots`).
- La corrida real exige kza-voice inactivo, el head cerrado y el lease de la
  raíz, y que `--confirm` sea el plan que se recalcula en ese momento.
- El id de decisión se valida sólo por formato, como en los demás comandos. El
  plan y el registro nombran además la autorización registrada (P7 = B,
  `u3b_0e5_retirement` y `r1e_questions_20261118`).
- Rechaza `runtime_admission_legacy_retire_l4_used` si existe algún registro
  `admission/launcher/uninstall-*.json`: L4 volvió a servir el estado
  congelado, y sus datos de ese período necesitan una decisión nueva del
  usuario.
- Rechaza `runtime_admission_legacy_retire_referenced:<qué>` si algo todavía
  apunta a los slots de b0866e8, e9669ec, 63ea8d3 o 2c5adb4 o a las bases de
  los dos respaldos:
  - un enlace bajo la raíz (`current` y `controller/current` incluidos);
  - un registro de `admission/` fuera de los históricos (`adoption/`,
    `launcher/` y los propios del retiro);
  - una unidad de usuario cargada (`ExecStart`, `Environment`,
    `EnvironmentFile`, `WorkingDirectory` o un bind de contenedor);
  - la configuración de logrotate de la consola: el mismo día, antes de la
    corrida de prueba, hay que quitar la estrofa de la consola vieja (ver el
    runbook);
  - el `settings.yaml` o el `.env` del estado servido.
- Un directorio que desaparece mientras la corrida de prueba recorre el estado
  vivo rechaza `runtime_admission_legacy_retire_scan` (no un traceback).
- No sigue enlaces ni cruza montajes: un montaje dentro de un destino rechaza
  (`runtime_admission_legacy_retire_mount`), y también una entrada del slot que
  no sea `runtime`, `state` ni uno de los tres informes como archivo regular
  (`runtime_admission_legacy_retire_slot_unexpected`).
- Antes del primer borrado escribe `admission/legacy-retirement/intent-<plan>.json`,
  inmutable, con la lista ordenada de destinos y los cuatro slots en
  `ineligible_slots` (e9669ec, b0866e8, 63ea8d3 y 2c5adb4). Desde ese momento
  L4 ya los rechaza. Si se corta, repetir el mismo comando con el mismo id y el
  mismo `--confirm` lo completa (una intención con otros destinos rechaza
  `intent_mismatch`). Al final deja
  `admission/legacy-retirement/retired-<plan>.json`, sin contenido: id y
  autorización de la decisión, conteos, nombres de las bases y la prueba de
  ausencia de cada destino.
- Es irreversible. Una segunda ejecución rechaza `already_retired`.

Después del retiro **no hay rollback no gestionado** (decisión Q3 = A):

- el estado de b0866e8, sus informes, sus veinte copias SQLite del 2026-10-29
  y los slots de 63ea8d3 y 2c5adb4 ya no existen, y los enlaces del contenedor
  de e9669ec cuelgan;
- `uninstall-launcher` rechaza los cuatro slots con
  `runtime_admission_uninstall_legacy_retired` y el nombre del registro;
- **`/home/kza/kza` no es un destino de rollback.** No es un slot, ninguna
  herramienta lo usa como destino y no se va a construir una: su unidad
  original corría desde `/home/kza/app`;
- la recuperación sólo avanza: una release gestionada nueva, o un rollback
  gestionado dentro del linaje gestionado. **Desde SC-2 ninguna de las dos
  existe hasta el modo release** (ver «Sin transiciones gestionadas después de
  R1 hasta el modo release»): después de este retiro no queda ningún rollback.
  La respuesta a SC-D5 registrada en esa sección es A: este paso no se
  posterga (sólo la opción B lo postergaría).

`/home/kza/kza` está congelado y sin retención desde el 2026-10-28, así que
guarda más datos vencidos que la imagen que se retira. Queda como copia, no
como destino, hasta una decisión Q5 aparte sobre `/home/kza/kza`. Lo mismo
vale para el respaldo `backups/release_20261028T144002Z_2c5adb4/`, que es de
`/home/kza/kza`.

Época del controlador (revisión 0e3 F14): desde R1 no se puede instalar un
controlador anterior a la época 2. Un rollback gestionado (no disponible desde
SC-2 hasta el modo release, ver «Sin transiciones gestionadas después de R1
hasta el modo release») podrá volver a slots anteriores, pero no el
controlador por debajo de la época 2.

Evidencia: `data/autonomous-roadmap-20261102/t040-t041/u3b-0e5-evidence.json`
(local, fuera de Git).

### Compactación de estados dueños (U3b-0c) — desarrollada, no ejecutada

Estado: `stage_runtime compact` está en la rama de desarrollo (U3b-0c), probado
en raíces de prueba, con ensayos compilados y con el systemd de usuario real
de `yo` para el escaneo de unidades. Nunca corrió sobre `/home/kza/releases`.
Pasos, fallos y códigos: [runbook](../runbooks/2026-11-14-owner-compaction.md).

Desde SC-1b, con el controlador en época 5 cada compactación nueva es una
transición del modo release (ver «Plan y start del modo release (SC-1b)»);
lo que sigue describe el procedimiento común y la compactación simple de la
época 4.

Qué cambia en el hueco de privacidad de arriba: el operador puede compactar la
raíz gestionada a un estado nuevo K (copia idéntica del head con las rutas
reubicadas en K) y, después de una corrida gestionada limpia de K, destruir
todas las demás copias que el controlador prueba haber creado: slots
gestionados anteriores (dueños o no), slots preparados y nunca activados,
copias del catálogo dentro de la raíz (también las que estén bajo `backups/`,
decisión D3 = A) y restos de `prepare` y `activate`. Un slot es miembro sólo
si contiene exactamente `runtime/` y `state/` con el `state/` del plan, y su
`runtime/` sólo los archivos de su `release.json` y los enlaces `config`,
`data`, `.env`; una copia del catálogo, sólo si cada entrada de primer nivel
está en sus extents o en el inventario de su preparación. Si no, queda entera
en la lista intacta (`unexpected_entries`, con los nombres). Un slot probado
sólo por un registro de staging se destruye únicamente con
`--retire-prepare-leftovers`. Los enlaces de los restos de `prepare` y
`activate` se borran como enlaces, sin seguirlos. Los miembros se destruyen
enteros. Con
eso, una conversación, un recuerdo o un perfil de voz borrados en el head
dejan de existir también en esas copias. La recomendación aparece en la salida
de `compact plan` (antes de SC-2 también en la de `delegate`) (al llegar a 6 dueños, o cuando el registro
de borrados del head muestra borrados pedidos que siguen vivos en un estado
retenido).

Límites que siguen valiendo (decisiones P5 = C y Q4):

- Sólo `unlink`: no se sobrescribe ni se corre `fstrim`. Los bloques liberados
  quedan en el disco hasta que se reutilicen; es un borrado lógico.
- VACUUM sólo sobre K. Desde SC-4c los almacenes de Chroma de K se
  reconstruyen con el SDK desde la lista cerrada (no se copian sus bytes),
  así que no traen páginas libres de H; lo que K' escriba después en Chroma
  sigue siendo opaco al SDK.
- No toca nada fuera de `/home/kza/releases` (`/home/kza/kza`, respaldos
  externos, snapshots del disco) ni nada dentro de la raíz sin prueba
  positiva: `backups/` (salvo los snapshots que el controlador catalogó; los
  respaldos del 2026-10-29 y el resto quedan intactos), `logs/`,
  `deployments/`, `dashboard-proxy/`, `controller/`, archivos de release, el
  contenedor de enlaces de e9669ec, el slot de b0866e8 (lo retira Q5), slots
  con entradas hechas a mano y restos `.prepare-*` sin registro. Una copia
  del catálogo fuera de la raíz impide compactar mientras exista; borrarla o
  no es decisión del usuario.
- El retiro recalcula el plan (una entrada nueva en la raíz rechaza, y un
  miembro movido o borrado después del plan rechaza
  `compaction_member_missing`; aceptarlo con `--accept-missing` lo deja fuera
  sin lápida ni registro de copia retirada: nada dice que se destruyó) y mide
  la cobertura contra H: lo que la corrida de K borró se cuenta aparte y no
  bloquea; datos nuevos sólo en miembros exigen reconocer el digest de la
  cobertura del retiro.
- El escaneo de unidades (P14) cubre las unidades nombradas donde estén, las
  enlazadas con `systemctl --user link`, las plantillas de los directorios del
  usuario (con líneas de continuación unidas, `$VAR` sustituidas desde el
  entorno del gestor, su `Environment=` y su `EnvironmentFile=`, que gana como
  en systemd) y los valores de `show-environment`, también los que systemd
  imprime como `$'...'`. En plantillas y entorno del gestor la regla es de
  subcadena: la raíz o un miembro en cualquier lugar del texto o de un valor
  rechaza, y cada trozo de ruta (cortado en espacios, caracteres del shell,
  `=`, `,` y `:`, así que vale dentro de `sh -c` y de listas) se juzga por la
  parte fija antes de `%i` o de un `$VAR` sin valor; un trozo que no se puede
  ubicar, o cualquier variable en el guion de un `sh -c`, rechaza salvo la
  tolerancia de U3b-0e4. En toda unidad un valor se corta también en `,` y en
  cada `=` (`podman --mount type=bind,src=<miembro>`), y un comodín de shell
  (`sh -c 'tar c <raíz>/slots/*/state'`, `%h/rel*/slots/%i`, o relativo con
  directorio como `releases/slots/*` desde el home) se juzga por su parte
  fija, componente por componente, con `[...]` como un carácter (R5-1 y su
  revisión); un patrón relativo sin directorio (`*/cache/*`) no. En las
  plantillas los escapes C (`%h/rel\x65ases`) se decodifican como en systemd
  255 y el guion de un `sh -c` se lee también sin comillas (R5-2). En una
  unidad tolerada, una variable sin valor seguida de la cola de un miembro
  rechaza. Una unidad es «de
  kza» por su nombre `kza*` o por una ruta de kza (`~/kza`, `~kza/kza`,
  `%h/…`, `$HOME/…`, `"$HOME"/…`, `${HOME:-…}/…`, la ruta absoluta aun con
  `//` o `/./`, un enlace del home como `/home/kza/app`, o un `kza` relativo
  desde el home); un `kza` dentro de otra ruta no cuenta. Igual en U3b-0e4.
- Una copia del catálogo en preparación con su proceso vivo (mismo pid y
  arrancado antes de la reserva) impide compactar
  (`compaction_copy_in_progress`); una abandonada aparece en el plan y nunca
  bloquea.
- Un abort también acepta un miembro propio que falta, o K movido a mano
  después de construirlo (`--member n --accept-missing`, sin lápida; K
  aceptado sigue donde quedó y es una copia completa de H). Sólo `walked`
  (registrado con la cuarentena vacía) prueba un borrado: `retiring` o
  `renamed` sin `walked` y el miembro fuera de su lugar rechaza
  `member_missing`. Un miembro aceptado que vuelve a su ruta rechaza
  `compaction_accepted_member_back` con número y ruta; no se borra solo. Si una re-comprobación tras una corrida de K detrás del abort
  rechaza `abort_revives`, el servicio queda detenido con H y K' intactos y la
  salida es un controlador corregido que siga desde K' (D4); el runbook lo
  detalla.
- Rollback: después de `complete` los slots anteriores ya no existen; sólo se
  avanza con una release gestionada, que desde SC-2 exige el modo release (ver
  «Sin transiciones gestionadas después de R1 hasta el modo release»). Antes de la verificación, `compact abort`
  vuelve a H. Después, `compact abort --to-h` vuelve a H sólo si durante la
  corrida de K no hubo borrados (Q6).

Duración y espacio (SC-0 y SC-0b, medidos en el servidor como `yo`, CPU, con
un estado sintético de producción de 154 MB: `events.db` 93 MB, `ambient.db`
60 MB). La copia (`compact plan` + `compact start`) tardaba 290-292 s; desde
SC-0b tarda 145-146 s (dos muestras: plan 10 s; start 135 s, con snapshot
42 s, materialización 74 s, autorización 3,5 s y activación 3,2 s). La
ganancia viene de la representación, no de quitar controles: la autoridad de
H se captura una vez por contenido y se reutiliza sólo si es idéntico todo lo
que la captura lee (el contenido de cada base SQLite con su `-wal`, `-shm` y
`-journal`, de cada registro de cuentas, de la configuración y de `.env`) y
la ruta y el tipo de cada entrada del árbol (cualquier cambio en eso, archivo
nuevo, renombrado o borrado, enlace simbólico o archivo especial vuelve a la
captura completa; el audio y los archivos opacos del SDK no se leen ni se
hashean, porque la captura no los lee), los plazos de cada texto de fuentes
se leen una vez, D5 compara digests SHA-256 por fila (las dos pasadas, antes
y después de VACUUM, siguen) y el manifiesto del snapshot se serializa una
vez por verificación. Diferidos: manifiesto sin filas (hoy guarda cada fila
como JSON, 377 MB, 2,4 veces el estado), VACUUM INTO dentro de K, menos
re-verificaciones del binding cerrado y hash en paralelo. Con SC-D2 = A
(10 minutos por transición) quedan unos 454 s para V boot + N + V stop +
arranque real.

Espacio (P9, corregido en SC-0b). El pico es el VACUUM de la base más grande
de K, con H en disco: backups del snapshot (S), manifiesto (M), estado de K
(S), runtime de K (R), journal de rollback del VACUUM (L, junto a la base) y
su base temporal (L, en el directorio temporal de SQLite: `SQLITE_TMPDIR`,
`TMPDIR`, `/var/tmp`, `/usr/tmp`, `/tmp`). `compact plan` y `compact start`
exigen libres en el sistema de archivos de la raíz 2·S + M + 2·L + R +
256 MiB, con M estimado en 3·S (calibrado: medido 2,45; no es una cota); si
el directorio temporal de SQLite está en otro sistema de archivos, una L pasa
a ese (L + 256 MiB). El trabajador vuelve a comprobar con tamaños reales dos
veces: dentro del snapshot, con el manifiesto ya serializado y antes de
escribirlo, M + estado de K + 2·L + 256 MiB; y después del snapshot, antes de
escribir K, estado de K + 2·L + 256 MiB (en ambos casos una L pasa al
directorio temporal si está en otro sistema de archivos; rechazo reanudable,
`runtime_admission_compaction_capacity`: el snapshot parcial se destruye al
reanudar). Así M sólo es una estimación para `compact plan`. Para el estado de
prueba: necesidad 1241 MB; términos reales 2·S + M + 2·L + R ≈ 887 MB;
crecimiento medido en el servidor 823-942 MB (5,3-6,1 veces el estado, por la
caída del espacio libre muestreada cada 10 ms; el servidor comparte el disco y
otros procesos escriben, así que una muestra pasa los términos por ~55 MB y
la absorbe el margen de 256 MiB). La regla anterior (2·S + 256 MiB = 577 MB)
no alcanzaba.
Hasta el retiro quedan ocupados el snapshot (S + M), K y su runtime: unos
697 MB, 4,5 veces el estado.

Compatibilidad: la época del controlador sube a 3. Mientras haya una
compactación pendiente, cualquier entrada al controlador que no sea `compact`
(prepare, activate, run, delegate, adopción) rechaza con
`runtime_admission_compaction_pending` (desde SC-2, `delegate` rechaza
siempre con `runtime_admission_delegation_retired`), y serve sólo deja pasar la corrida de
verificación armada o las corridas de un `allow-run` vigente; desde que
empieza un abort o un retiro, ninguna (`compaction_abort_pending`,
`compaction_retire_in_progress`). Si el head cambió igual después de
`abort_started`, el abort exige repetir Q6 con `--to-h` antes de seguir
(`compaction_abort_recheck_required`), y un abort o retiro reanudado termina
cada miembro por su identidad registrada, sin registrar el final mientras
quede algo en la cuarentena.
`install-controller` (época 4 o posterior desde SC-2, nunca menor que la instalada) sí
entra, también con un log inválido: es la salida de una compactación trabada
(D4 = A). Desde SC-1a (época 5, sin release) la primera instalación de época 5
rechaza cualquier log pendiente o ilegible; las siguientes rechazan un log
ilegible o un log simple, y con un log del modo release entran sobre el head
cerrado (SR-1) o sobre el H o el K abiertos de ese log (ver «Controlador de
época 5 (SC-1a)»). La recuperación automática R-b queda apagada mientras tanto. No
reinicies el servidor durante una compactación.

Épocas (D1 = A): R1 se construye desde un árbol que ya contiene U3b-0c y,
desde SC-2, la delegación retirada, así que ningún controlador de época 2 o 3
vive en la raíz gestionada; la adopción exige `controller_epoch == 4` (igual a
la época del código que la ejecuta). SC-2 subió la época a 4 sin cambiar el
formato de slot (`FORMAT_EPOCH` 1, `LAUNCHER_CONTRACT` 3): los slots anteriores
siguen corriendo bajo un controlador de época 4. Desde SC-2, `release.json`
declara también `schema_contract` (`SCHEMA_CONTRACT_EPOCH`, hoy 1; un archivo
anterior se lee como 0): el contrato de esquema y vocabularios queda fijado
por un test, y el modo release sólo moverá entre archivos con el mismo. Una copia vieja de `stage_runtime` (en un slot o
en `controller/<commit>`) puede correrse directamente y no conoce la
compuerta: sólo se usa `controller/current`. Desde la época 3, la entrada al
controlador y `prepare` rechazan cuando la época instalada es mayor que la
propia.

Evidencia: `data/autonomous-roadmap-20261102/t040-t041/u3b-0c-evidence.json`
(local, fuera de Git).

### Identidad de destinos SQLite y temporales propios (F4/F10, 2026-11-14)

Modelo de amenaza: otro proceso con el mismo uid borra y recrea un archivo en
una ruta que KZA ya comprobó. En ext4 el número de inodo liberado pasa enseguida
a la siguiente asignación, así que `(st_dev, st_ino)` identifica un archivo sólo
mientras su inodo sigue asignado.

- Destinos SQLite nativos (privacidad, sesiones, rutinas, eventos, sugerencias,
  objetivos históricos, listas y recordatorios): en Linux, el coordinador de
  salidas abre cada destino con `O_PATH | O_NOFOLLOW` antes de que el plan lo
  observe, y el plan debe nombrar ese inodo. Un destino que crea su primera
  apertura se fija al registrar `active`. Las reaperturas y los objetivos
  históricos comparan contra ese inodo retenido. Todos los descriptores se
  cierran al cerrar la época, envenenada o no, después de drenar el ejecutor de
  disco del coordinador. Desde ese momento no se entrega ningún pin nuevo: una
  llamada nativa posterior (por ejemplo tras un cierre con `epoch_incomplete`,
  que no envenena) se rechaza con `output_registration_native_pins_closed` y
  envenena la época. Un objetivo histórico cuyo pin ya se cerró se rechaza con
  `destination_changed`; nunca se acepta por `(st_dev, st_ino)` sin pin.
- Si no se puede abrir el pin por algo distinto de un destino ausente (por
  ejemplo `EMFILE`/`ENFILE`), el rechazo es
  `output_registration_native_pin_unavailable`, no `destination_changed`.
- Por qué `O_PATH`: cerrar un descriptor `O_PATH` no suelta los locks POSIX de
  SQLite del proceso (el kernel omite `locks_remove_posix` para
  `FMODE_PATH`). Un descriptor `O_RDONLY` extra sí los suelta, en Linux y en
  macOS, por eso los publicadores JSON/YAML usan `O_RDONLY` y SQLite no.
- macOS no tiene `O_PATH` y APFS no reutiliza identificadores de archivo, así
  que allí no se fija nada.
- Efecto visible: hasta el cierre de la época, un destino borrado por otro
  conserva sus bloques en disco, y cualquier proceso del mismo uid puede leerlo
  por `/proc/<pid>/fd`. La época dura lo que dura el servicio (días o semanas),
  no un plazo fijo. Ningún dueño nativo borra sus propios destinos, y los
  retiros del controlador corren con el servicio detenido.
- Temporales propios (`mkstemp`): `stage_runtime`, `managed_launcher`,
  `build_runtime`, catálogo y observaciones de copias, derivación de payload,
  cooldown, contextos, `audio_health` y diagnósticos retienen un `dup` del
  descriptor hasta limpiar. Sólo borran el nombre si sigue siendo ese inodo, y
  si hay otro archivo en ese nombre lo dejan. Si el `dup` falla (`EMFILE`), el
  descriptor de `mkstemp`, todavía abierto, hace de pin: el temporal vacío se
  borra igual y el error se propaga. Ayudante común en
  `src/core/own_temporary.py`; `stage_runtime` y `build_runtime` tienen copias
  propias porque se cargan por ruta.
- Huecos conocidos (comparaciones que siguen sin pin):
  - un destino que crea su primera apertura no tiene identidad entre la creación
    de SQLite y el registro `active`; si se sustituye en esa ventana, el pin se
    toma sobre el sustituto;
  - un objetivo histórico ausente en el refresco del registro no tiene pin ni
    identidad (`allow_absent`): un archivo que aparezca después se acepta hasta
    que cambie el digest del registro;
  - `observe()` deduplica observaciones físicas por un `lstat` sin pin; una
    reutilización sólo suprime una observación del journal y no concede
    autoridad;
  - las identidades de `epoch.json` y del namespace del journal (fuera de F4);
  - la recuperación de `diagnostic_store` después de un corte
    (`_cleanup_staged`) borra todavía por nombre, porque el descriptor no
    sobrevive al proceso.

Evidencia: `data/autonomous-roadmap-20261102/t040-t041/f4-f10-evidence.json`
(local, fuera de Git).

## Limpieza del checkout anterior — pendiente

La inspección del 2026-10-25 encontró:

- `/home/kza/app` apunta a `/home/kza/kza`; `kza-voice.service` ejecuta desde
  allí `/home/kza/app/.venv/bin/python -m src.main`.
- El Quadlet versionado del índice monta `/home/kza/app`, y el indexador lee
  `src/**/*.py`. Hay que confirmar los montajes activos desde la cuenta `kza`
  y trasladar la lectura al checkout de `yo` antes de retirar los fuentes.
- El LLM usa un modelo en `/home/kza/kza/models`; ese directorio debe
  conservarse junto con `data`, `config`, `.env`, `logs` y el entorno Python.
- `yo` no tiene escritura en el checkout ni acceso a la sesión systemd de
  `kza`; `sudo` sin contraseña y SSH con la clave actual tampoco están
  disponibles. La instalación requiere una sesión administrativa autorizada.

La publicación del 28 de octubre completó la instalación separada y el cambio
de unit, conservando las rutas antiguas. Para retirar material de desarrollo,
manteniendo el servidor bajo `kza`, quedan estos pasos:

1. Inspeccionar las units, timers y montajes reales desde la sesión de `kza`.
   Registrar todas las referencias a `/home/kza/app`, `/home/kza/kza` y scripts
   `.py`, incluidos los jobs de entrenamiento y watchdog.
2. Migrar el índice de código para que lea el repositorio de `yo`, y actualizar
   las invocaciones de scripts que necesiten su variante `.pyc`.
3. Completar la aceptación de voz, micrófonos, efectos HA, rutinas, privacidad
   e índice. Mantener el checkout original para rollback mientras se valida.
4. Retirar únicamente los fuentes y material de desarrollo del checkout
   anterior, después de comprobar su respaldo bajo `yo` y en GitHub. También
   inventariar `/home/kza/bench-ambient`, `.claude/worktrees` y
   `src.bak-20260424-1741`, que contienen copias históricas de código.

Esta limpieza todavía no se ejecutó. El pipeline principal usa bytecode del
slot; el índice y los recursos externos todavía dependen de rutas antiguas.
Conservar `/home/kza/kza` y `/home/kza/app` hasta cerrar esas dependencias y la
ventana de rollback. El desarrollo continúa bajo `yo`.

## Cómo corre producción (server `192.168.1.2`, usuario `kza`)

- **kza-voice.service** — pipeline de voz **nativo** (NO contenedor), systemd `--user` en
  `/home/kza/.config/systemd/user/`. Es la **excepción R10 documentada** del contrato del
  servidor (acceso USB ReSpeaker + MA1260 serial + presupuesto <300ms hacen al contenedor
  inviable). Ver `docs/SERVER_CONVENTIONS.md` § Excepciones a R10.
- **llama-server :8101** (Qwen2.5-7B Q4_K_M, cuda:1) — fast-path NLU, bajo `kza`.
- **Gateway LiteLLM :8200 → MiniMax cloud** — reasoner slow-path, bajo `infra` (no lo
  administra KZA; consumir con virtual key).
- **ChromaDB** — hoy in-process dentro de kza-voice. Existe el Quadlet
  `kza-chroma.container` para migrarlo a :9500, pero NO está operativo: la auditoría
  2026-05-30 lo encontró en crash-loop (monta `chroma` en vez de `chroma_db` —
  fix pendiente; ver `docs/SERVER_CONVENTIONS.md` § deudas 2026-05-30).
- Secrets en `/home/kza/secrets/` (`.env`, `*-api-key.env`), inyectados vía
  `EnvironmentFile=` del unit.

## Cómo se deployaba — histórico, sustituido por artefactos

Flujo git local ↔ server ↔ GitHub histórico documentado en
[`ECOSISTEMA_LOCAL_SERVER.md`](ECOSISTEMA_LOCAL_SERVER.md): el server es repo git y
deploya **in-place** (el código que corre es el working tree), no pushea; la laptop es el
puente (`scripts/kza-push` / `kza-sync`). Restart manual del service cuando el cambio lo
requiere — chequear VRAM libre antes (`nvidia-smi`; preflight integrado avisa si <1500MB).

## Qué NO es producción

- **`docker/` (Dockerfiles + services) — EXPERIMENTAL, sin paridad** con el monolito
  canónico (`python -m src.main`). Ver `docker/README.md` para el detalle de gaps.
  No deployar producción con esto.
- **docker-compose** — el `docker-compose.yml` de la raíz del repo pertenece al modo
  experimental de `docker/` (mismo estatus: sin paridad, no producción); las menciones
  en `docs/research/` y `docs/plans/` de 2026-02/03 son propuestas históricas. Nada de
  compose corre en producción.
- Units de sistema en `/etc/systemd/system/` — prohibido por contrato (todo va en
  systemd `--user`).
