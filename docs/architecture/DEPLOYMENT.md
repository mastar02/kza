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

El próximo modelo de despliegue transferirá a `kza` únicamente el artefacto
compilado y sus recursos de ejecución, construidos y probados bajo `yo`.
Secretos y datos persistentes permanecen fuera del artefacto y de Git. El
proyecto es Python: `scripts/build_runtime.py` construye un paquete de bytecode
CPython 3.13 sin fuentes `.py`. El ensayo offline de instalación, promoción y
rollback está en `scripts/stage_runtime.py` y el
[plan de H0](../plans/2026-10-25-h0-baseline-contracts.md). Su validación usa
staging bajo `yo`; la integración de dependencias, modelos, units y estado real
se completa para la release. No se cambió el servicio durante el rescate.

La sección siguiente describe el despliegue existente. El flujo histórico de
edición en producción y puente desde la laptop queda retirado para trabajo
nuevo. Consultar `AGENTS.md` en la raíz para las instrucciones operativas.

## Construcción del artefacto

El candidato de estabilización liga el dashboard a `127.0.0.1:9500` detrás del
proxy del servidor. Todo endpoint HTTP/WS privado requiere `KZA_DASHBOARD_TOKEN`,
incluidos health y observabilidad. La variable de ejemplo está vacía; el valor
real permanece en el entorno privado de ejecución. La shell de login y los
assets son públicos, sin datos del hogar. No se expone OpenAPI por una ruta
automática. Los clientes de diagnóstico deben enviar Bearer o sesión válida.

En la única release, el proxy debe preservar Host y establecer correctamente el
esquema externo para que se validen Origin y la cookie Secure en HTTPS. Sólo se
confía en forwarding desde el proxy local; no se toman headers de cualquier
cliente como prueba de identidad. Las pruebas ASGI comprueban la frontera, pero
la configuración instalada del proxy y la unit se validan en H6 antes de
promover. Este cambio de configuración versionada no altera el servicio actual.

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

## Limpieza del checkout de desarrollo — con la próxima release

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

Secuencia para la próxima publicación, manteniendo el servidor bajo `kza`:

1. Inspeccionar las units, timers y montajes reales desde la sesión de `kza`.
   Registrar todas las referencias a `/home/kza/app`, `/home/kza/kza` y scripts
   `.py`, incluidos los jobs de entrenamiento y watchdog.
2. Instalar el artefacto en una release separada, sin Git ni fuentes `.py`.
   Conectar sus rutas de configuración, datos, modelos, logs y dependencias
   con las ubicaciones persistentes existentes.
3. Migrar el índice de código para que lea el repositorio de `yo`, y actualizar
   las invocaciones de scripts que necesiten su variante `.pyc`.
4. Cambiar la unit de voz a la release y comprobar voz, frontend, micrófonos,
   rutinas e índice. Mantener el checkout original para rollback mientras se
   valida. Los diez tests fallidos del rescate todavía requieren revisión.
5. Retirar únicamente los fuentes y material de desarrollo del checkout
   anterior, después de comprobar su respaldo bajo `yo` y en GitHub. También
   inventariar `/home/kza/bench-ambient`, `.claude/worktrees` y
   `src.bak-20260424-1741`, que contienen copias históricas de código.

Esta secuencia todavía no se ejecutó: los servicios siguen usando el checkout
original. Los permisos administrativos se necesitarán al publicar esa release;
el desarrollo actual continúa bajo `yo`. Conservar los archivos de ejecución y
el estado persistente de `/home/kza/kza` hasta completar esa sustitución.

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

## Cómo se deployaba — pendiente de reemplazo por artefactos

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
