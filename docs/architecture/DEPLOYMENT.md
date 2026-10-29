# Deploy de KZA — fuente de verdad

> Si otro documento de este repo contradice a este, gana este (y ante dudas de
> contrato compartido, `docs/SERVER_CONVENTIONS.md` / Notion pág. 8).

## Desarrollo y despliegue acordados — 2026-10-25

El desarrollo, los tests, la construcción y Git se ejecutan como `yo` en
`/home/yo/projects/kza`. `kza` es la cuenta administrativa y de ejecución del
servidor (UID 1000, grupo `sudo`); no se usa para desarrollar ni compilar.

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
- `kza-voice.service` conserva su unit original y usa el drop-in `release.conf`.
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
reemplaza un archivo de salida existente. El mismo commit produce el mismo
archivo con la misma versión de CPython.

Es bytecode para CPython 3.13, no un ejecutable nativo ni una barrera contra
descompilación. Requiere el intérprete y las dependencias del entorno de
ejecución. Ver [py_compile](https://docs.python.org/3.13/library/py_compile.html)
y [SourcelessFileLoader](https://docs.python.org/3.13/library/importlib.html#importlib.machinery.SourcelessFileLoader).

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
