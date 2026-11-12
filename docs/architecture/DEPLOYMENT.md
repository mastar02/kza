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

Está activo `release_20261029T174830709997Z_63ea8d3`, construido desde
`63ea8d319b50506cc37ba73df1e41c87a41054cf`. La actualización del 29 de octubre
ordena las muestras de Usuarios / Voces por fecha descendente y verifica la
paginación. El operador publicó el artefacto compilado solicitado por el usuario;
`yo` conserva acceso de diagnóstico, sin shell operativa en `kza`.
Esta instalación no acepta H6 ni cierra el roadmap. La última suite completa
mantiene dos fallos conocidos. La observación posterior confirma captura en Hall
y Escritorio; Living y Cocina siguen sin captura útil. HA conserva `auth_error`.

- Runtime: `/home/kza/releases/current` apunta al subdirectorio `runtime` del
  slot instalado. Se verificaron los 360 archivos del manifiesto, con 324 módulos
  compilados y sin fuentes `.py`.
- Estado: `state`, hermano de `runtime`, es un contenedor privado cuyos enlaces
  conservan las rutas canónicas de configuración, datos, entorno, modelos y logs
  del slot `release_20261028T145442Z_b0866e8`. No se movieron ni reemplazaron datos:
  los registros de retención conservan sus localizadores absolutos. El directorio
  de trabajo es el nuevo contenedor; `runtime/config`, `runtime/data` y
  `runtime/.env` enlazan a él. No retirar el estado del slot anterior.
- `kza-voice.service` conserva su unit original y usa el drop-in `release.conf`
  (el lanzador gestionado U3b-0d lo reemplazará en R1; ver «Lanzador gestionado (U3b-0d)»).
  Ejecuta `/home/kza/kza/.venv/bin/python -B -P -m src.main`, con
  `PYTHONPATH=/home/kza/releases/current`. El venv y los modelos existentes
  permanecen externos; no se instalaron dependencias ni se reasignaron GPUs.
- Actualización: sólo cambió `WorkingDirectory` en el drop-in existente; el
  selector de código continúa siendo `PYTHONPATH=/home/kza/releases/current`.
  Se reinició únicamente voz. El proxy y la rotación de consola conservaron su
  ejecución. Diez bases SQLite tienen respaldo consistente en
  `releases/backups/release_20261029T174830709997Z_63ea8d3/`.
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
  sustituya. Ojo: la release instalada (slot activo 63ea8d3) no tiene
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

Está preparada, aún sin instalar, la mejora de carga de muestras de Voces desde
`e9669ec1edb101ca6b190ef8df09314e1ac63562`. Cambia únicamente dos assets de la
aplicación: hace visible el reproductor al seleccionar una muestra, muestra el
progreso y conserva el audio seleccionado al repetir la selección. El diagnóstico
por HTTPS desde el servidor midió 5–11 ms para ocho muestras; no reproduce la red
ni Chrome en Windows del usuario. Ver la
[evidencia y límites](../research/2026-10-29-voice-loading.json) y el
[comando de publicación](../runbooks/2026-10-28-compiled-runtime.md#mejora-de-carga-de-muestras--preparada-2026-10-29).

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
producción actual b0866e8 (o a 63ea8d3) queda rechazado; sólo una activación no
gestionada, sin head ni autoridad, los acepta todavía. El rollback posterior
debe apuntar a una release que incluya el par.

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

`stage_runtime --root R --offline delegate <slot>` delega al slot copiado todo
lo que el head cerrado tiene. Incluye sus destinos de borrado propios con rutas
y los que heredó, aunque no tengan rutas. Desde U3b-0b (en desarrollo, sin
release) admite varios saltos. Así funcionan la segunda release gestionada, un
rollback y el roll-forward posterior.

Cada registro de `admission/delegations/` conserva el esquema v1 y nombra al
dueño físico de las rutas. El código 0a de slots anteriores los acepta. Al lado,
`admission/delegation-chains/<capability>.json` guarda la cadena: dueño primero,
emisor al final. Cada salto incluye su binding, la prueba de cierre embebida y
el registro que le dio el destino. El controlador sólo honra registros con esa
cadena y deriva cada cadena del head verificado, nunca del llamador.

Antes de escribir nada, rechaza estos casos:

- volver a un estado que ya fue dueño o salto: `runtime_admission_delegation_cycle`;
- cadenas faltantes, alteradas o truncadas: `runtime_admission_delegation_chain`
  o `runtime_admission_delegation_chain_missing`;
- una concesión que omite un destino heredado: `runtime_admission_delegation_incomplete`;
- una colisión con otra cadena: `runtime_admission_delegation_conflict`;
- más de 8 estados dueños distintos (con 8 todavía se concede): `runtime_admission_delegation_owner_limit`.

La API devuelve una capability por dueño. La sonda de `activate` ahora también
corre el `validate_target_ownership` del candidato, así que un candidato que
recalcularía distinto a un dueño se rechaza antes de mover el head.

Límites:

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

  El límite de la sonda (60 s por defecto) no cambia. Con 8 dueños de ese
  tamaño, la sonda 0b ronda 10 s. Con estados de unos 2 GB, una sonda 0b con 8
  dueños ronda 40 s. Una anterior a 0b, con varios destinos por dueño, pasa
  de 60 s: hay que medir con una copia real
  (`tests/mocks/probe_cost_harness.py --multi-hop`) antes de la release y
  ajustar `--probe-timeout` sólo con esa medición.
- No hay compactación. Una cadena admite hasta 8 dueños distintos. Cada
  transición gestionada (release, rollback o roll-forward) suma el dueño
  anterior, así que el estado número 10 de una cadena, que necesitaría 9
  dueños, se rechaza con `runtime_admission_delegation_owner_limit`. Eso pasa
  en la novena transición desde el primer estado gestionado. Decidir antes
  qué hacer es del usuario.
- Volver a ejecutar un salto intermedio no revoca nada. Su prueba de cierre
  embebida debe venir de una ejecución de su propio slot y estado, y se
  comprueba contra el recibo inmutable.
- Borrar en un dueño puede dejar el head en UNKNOWN (hallazgo F8, heredado de
  0a y sin corrección local). Si el hijo del head borra, mediante un destino
  heredado, una fila que forma parte de la proyección protegida del dueño, el
  dueño deja de coincidir con su binding. Un ejemplo es un evento con su plazo
  de texto acortado. El cierre del hijo falla con
  `runtime_admission_delegation_missing` y el head queda UNKNOWN. Borrar
  contenido ordinario no tiene ese efecto. Corregirlo requiere volver a
  vincular el dueño bajo la autoridad del head, un diseño pendiente.
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
- fuera de la raíz de admisión: el estado de producción de b0866e8 (`/home/kza/kza`), los respaldos de `/home/kza/releases/backups/` y cualquier snapshot del disco.

Los embeddings de esos recuerdos en los índices semánticos de los estados dueños sí se borran, porque esos índices son destinos con ruta.

Consecuencia: si una persona pide «olvidá mi voz» u «olvidá esto», su perfil de voz o su recuerdo desaparece del estado activo, pero sigue en cada slot anterior y en cada copia hasta que esa copia se destruya. Con la compactación (U3b-0c, en diseño; requiere antes la unidad del lanzador gestionado), esas copias se destruyen al compactar. Mientras tanto pueden ser hasta 5 slots anteriores si se compacta al llegar a 6 dueños. El borrado de la compactación es sólo unlink: no sobrescribe. Las copias fuera de la raíz no las alcanza ningún mecanismo actual.

Hasta que exista una herramienta, retirar a mano una copia vieja requiere una decisión explícita del usuario. Nunca borres el estado del slot activo ni el de un dueño que el head todavía necesita (ver «Todo estado dueño debe quedar intacto»).

Versiones del controlador: en cuanto exista un registro escrito por 0b,
`delegate`, `activate` y `run` deben ejecutarse con un `stage_runtime` 0b o
posterior. Tras un rollback, no uses como controlador el `stage_runtime` del
slot restaurado si es anterior. Un controlador anterior a 0b rechaza el
multi-salto con `runtime_admission_delegation_one_hop_only`. Si concede un
primer salto, escribe un registro sin cadena. Un controlador 0b lo rechaza
(`runtime_admission_delegation_chain_missing`), y ese head ya no se puede
delegar hacia adelante.

Fallos a mitad de una concesión (orden de escritura: cadenas, registros,
política, manual head, CAS del head):

- Falla al escribir cadenas o registros: quedan archivos inmutables inertes que
  ninguna política lista. La política, el marker y el head no cambian. Para
  recuperar, repetí `delegate <slot>`: reescribe archivos idénticos y termina.
- Falla después de escribir la política (por ejemplo, en el manual head): la
  concesión ya vale y `activate` la acepta, pero faltan el registro de manual
  head y el aumento de generación del head. Para recuperar, repetí
  `delegate <slot>`: valida, reescribe lo idéntico y completa el CAS.
- Quedaron restos y después el emisor volvió a ejecutarse (su prueba de cierre
  cambió): ese destino queda en conflicto permanente
  (`runtime_admission_delegation_conflict`). Para recuperar, prepará un slot
  nuevo desde el estado del head y delegá a ése.

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
- Unit: `Type=notify`, `NotifyAccess=main`, `KillMode=mixed`,
  `RestartPreventExitStatus=69 78`, `Restart=on-failure`,
  `StartLimitBurst=5` en 300 s. Es un desvío registrado en
  `SERVER_CONVENTIONS.md` (excepción R10 #6).
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
  `TimeoutStartSec` no caiga a mitad de una adopción. `--timeout-start` se
  dimensiona con un R-b medido en el server (L2, L6).
- **Recuperación por arranque**: si el run quedó del boot anterior (reinicio o
  corte de luz), hay una recuperación por `boot_id` y no gasta el cupo de 24 h.
- **Operador** (`UNKNOWN` sin adopción automática): primero
  `stage_runtime --root /home/kza/releases --offline adopt --dry-run`, que
  imprime el sha del head, si R-a sigue siendo posible (recibo, registro y
  manifiesto) y sólo conteos. Es de sólo lectura: no abre nada del estado
  para escribir y corre los chequeos de bases sobre una copia privada (con su
  `-wal`, sin `-shm`), así que no puede romper un R-a pendiente. Después
  `adopt --confirm <sha> --confirm-regressions <familias>`, con las familias
  exactamente como las informó el dry-run.
- Un head recuperado no puede delegar hasta un cierre limpio posterior.

Instalación en la ventana de R1 (bajo `kza`, sin sudo; el servicio detenido):

1. Bajo `yo`: construir la release con `scripts/build_runtime.py`. Su
   `release.json` declara `launcher_contract`, `format_epoch`,
   `receipt_state` y `controller_epoch`, tomados del propio
   `scripts/stage_runtime.py` de esa revisión.
2. Transferir el archivo y `systemd/kza-voice.service.d/managed.conf.template`.
3. Extraer el archivo en un directorio temporal y correr
   `<tmp>/scripts/stage_runtime.pyc --root /home/kza/releases --offline install-controller <archivo>`.
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
     `--stop-grace`, `--timeout-start`, `--timeout-stop`, `--watchdog-sec`,
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
Ambos pasan a la lista de conservación de la unidad de retiro Q5.

Volver a b0866e8 o 63ea8d3 (L4 = A):

```
stage_runtime --root /home/kza/releases --offline uninstall-launcher \
  --unit-dir ~/.config/systemd/user/kza-voice.service.d \
  --user-decision <id> --phrase 'revivir datos borrados' --legacy-slot <slot>
```

- Requiere la unidad detenida y verifica el commit y el manifiesto del slot
  legacy.
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

Rollback gestionado: un slot destino necesita `launcher_contract >= 3` y
`format_epoch` igual o mayor que el mínimo del controlador. Si no, `serve`
lo rechaza con 78 (`runtime_admission_launcher_contract_missing` o
`runtime_admission_rollback_format_closed`). Con un controlador instalado
(`admission/controller.json`), `stage_runtime activate` aplica los mismos
enteros y rechaza antes de mover el head o `current`, así un activate del
operador no deja un arranque que sale con 78. La API acepta
`launcher_checks=False` sólo para ensayar ese rechazo de `serve`; la CLI no
lo expone. `run()` sigue aceptando slots anteriores para los ensayos de
linajes viejos.

Evidencia: `data/autonomous-roadmap-20261102/t040-t041/u3b-0d-evidence.json`
y, para las correcciones de la revisión independiente,
`u3b-0d-fix-evidence.json` (local, fuera de Git).

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
