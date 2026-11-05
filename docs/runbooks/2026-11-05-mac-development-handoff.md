# Continuar el desarrollo de KZA en Mac

Traspaso del desarrollo solicitado por el usuario al cerrar T023-F.
El entorno Mac queda pendiente de preparar y verificar; esta entrega conserva
los resultados obtenidos en Linux y el mismo objetivo del roadmap.

## Punto de reanudación

El usuario pidió terminar la unidad en curso, guardar el avance y retomar el
mismo goal desde una sesión nueva en Mac. El objetivo sigue en
`docs/plans/2026-11-02-autonomous-roadmap.md`; el estado operativo está en
`docs/plans/2026-11-02-autonomous-roadmap.json` y el historial en
`docs/runbooks/2026-11-02-autonomous-execution.md`.

| Dato | Estado que debe quedar en el cierre |
| --- | --- |
| Rama | `recovery/production-2026-10-25` |
| Última entrega verificada antes de F | `be32294`, T023-E: 8294 PASS y una xfail conocida; 1010 archivos, incluidos 37 scripts, sin cambios durante esa full |
| Unidad entregada | T023-F: autoridad y retiro de captura/inferencia/reporte de `diagnose_wakeword` |
| Commit de código F | `38f3912d5272be408f9d27afb74a298a54d2b5ac`; comprobar su publicación con el historial de la rama y el registro de cierre |
| Full F y evidencia pública | 8341 PASS y una xfail conocida; 1013 archivos, incluidos 37 scripts, sin cambios; XML 851.448 s. Evidencia: [captura de diagnóstico](../research/2026-11-05-wakeword-diagnostics-capture-authority.json) |
| Commit de tracking y handoff | El commit que contiene este archivo y las actualizaciones del roadmap; obtener su SHA con `git log -1 --format=%H -- docs/runbooks/2026-11-05-mac-development-handoff.md`. El mensaje de cierre confirma el push y el checkout limpio. |
| Estado del goal al salir | Pausa solicitada por el usuario después de esta entrega, con el objetivo incompleto. Confirmar el estado del mismo goal al retomar; no crear otro objetivo ni marcarlo completo. |
| Procesos de desarrollo pendientes | La full, las tandas dirigidas y Chromium de F terminaron; el cierre exige consumir también las comprobaciones documentales y los pushes. No hay una validación de audio que la sesión nueva deba duplicar. |

En la lectura previa al cierre, T023 y T022 están `in_progress`, T024 está
`planned` y los 21 criterios siguen `not_evaluated`. La migración de equipo no
acepta criterios ni modifica T020/P1. T047/NEXO ya figura `verified_local`:
conservar el diseño integrado, el backend real y la separación entre
voz, persona, cuenta, roles, consentimiento y administración.

## Autorización y entorno local

La petición explícita de migrar el desarrollo a Mac autoriza el checkout y el
entorno de desarrollo locales del Mac. Las instrucciones `yo` y
`/home/yo/projects/kza` describen el servidor de desarrollo Linux; no hay que
crear ese usuario ni reproducir esa ruta en Mac. Usar la cuenta habitual sin elevar
privilegios y la ruta local que el usuario elija; registrar ambas al retomar.

Esto no cambia la cuenta de producción `kza`, su despliegue, sus servicios,
sus datos ni la autorización de releases. No desarrollar en `/home/kza/kza` o
`/home/kza/app`, reiniciar producción, instalar allí ni hacer despliegues como
parte de esta migración. AGENTS/CLAUDE recogen esta excepción acotada para que las reglas específicas
del servidor Linux no impidan el desarrollo Mac autorizado.

## Preparación en Mac

1. Elegir el usuario y directorio locales. El repositorio es `https://github.com/mastar02/kza.git`.
   Clonar la rama `recovery/production-2026-10-25` desde ese remote; si ya existe un checkout, proteger sus cambios antes
   de actualizarlo. Comparar el HEAD con el último commit publicado de la rama que contenga este documento.
2. Registrar macOS, arquitectura real y versión de CPython. El proyecto y CI
   usan CPython 3.13. Crear un entorno nuevo en el Mac; no copiar el venv Linux,
   binarios compilados ni `data/project-assessment-20261102/venv`.

   ```sh
   python3.13 -m venv .venv
   . .venv/bin/activate
   python --version
   python -m pip --version
   ```

3. Resolver las dependencias para esa arquitectura antes de ejecutar la suite.
   `requirements-test.txt` es la lista usada por CI, pero fija
   `torch==2.12.0+cpu` con un índice y medición de wheels Linux. Su propio
   comentario distingue darwin/arm64. No está validada aquí una instalación
   directa sin cambios en macOS: comprobar disponibilidad y documentar cualquier
   adaptación de plataforma sin reemplazar el pin Linux ni recortar dependencias
   a ojo. `tests/conftest.py` requiere imports reales de `torch` y `soundfile`;
   un `MagicMock` global no equivale a preparar el entorno.
4. `requirements.txt` incluye dependencias de ejecución/modelos; no usar el
   instalador Ubuntu como instalador Mac. Resolver primero el entorno de tests,
   registrar las versiones efectivas y conservar cualquier fallo de instalación
   como dato de portabilidad. No descargar modelos ni copiar credenciales,
   grabaciones, embeddings, bases de datos o el árbol privado `data/` mediante Git.
5. Confirmar que Serena apunta al checkout Mac correcto y leer sus instrucciones
   iniciales y `mem:core` cuando estén disponibles. Si ctx/RAG/Serena no están
   preparados allí, informar el límite y usar lectura enfocada; no consultar
   silenciosamente otro checkout ni crear índices/túneles como efecto lateral.

No se conoce todavía la arquitectura, usuario, ruta, Python instalado ni
disponibilidad de wheels del Mac. Las herramientas Git y del asistente deben
autenticarse por sus flujos propios; este documento no transporta secretos.

## Comprobación y próxima unidad

Primero leer las evidencias públicas de F y el resultado de su full. Los
recibos privados bajo `data/` se conservan en el entorno Linux y no se presume
que existan en un clon nuevo. Una ruta ausente en Mac no invalida ni reemplaza
el resultado histórico: registrar qué material está disponible.

Con el entorno Mac instalado y registrado, las pruebas del cliente son un
primer conjunto acotado, sin iniciar el sistema ni los CLI de audio:

```sh
python -m pytest tests/unit/tools/test_calibration_authority_client.py \
  tests/unit/audio/test_native_capture_authority_client.py
```

Estos comandos son instrucciones futuras, no resultados obtenidos en Mac.
Conservar el primer resultado y distinguir incompatibilidad de plataforma,
fixture y defecto del producto. La evidencia de CUDA, USB, audio físico y SDK
Linux sigue ligada a su entorno; una prueba simulada o un skip en Mac no la
reemplaza. El harness Chromium actual usa `chromium` y un perfil snap de Linux;
su ejecución en Mac necesita una adaptación explícita del fixture, sin cambiar
oráculos para obtener un verde. Los screenshots móviles previos son sólo
observación y no aceptan iPhone/C07.

La propuesta siguiente es **T023-G/MultiMicCapture-zone-setup-live-inputs**:
vincular `zone_setup test-mic`/`test-detection` y las APIs públicas
`MultiMicCapture` a una autoridad administrativa explícita, con dueño previo de
cada stream y retiro real de callbacks, colas y reportes antes del ACK. El plan portable está en [continuación de MultiMicCapture](../plans/2026-11-05-multimic-capture-continuation.json),
con estado `proposed_read_only_no_runtime_no_RED`. Su copia original privada
es `data/autonomous-roadmap-20261102/t023/f/next-boundary-plan.json`. Todavía no se implementó G
ni se ejecutó su baseline. Antes de hacerlo, confirmar por lectura enfocada el
constructor de `ZoneManager`; conservar separados listado de dispositivos,
reproducción sintética y controles MA1260. No empezar G durante este cierre.

## Prompt para la sesión nueva

> Retomá el mismo goal de KZA de
> `docs/plans/2026-11-02-autonomous-roadmap.md`, pausado por la migración a Mac.
> Leé primero AGENTS.md, CLAUDE.md y
> `docs/runbooks/2026-11-05-mac-development-handoff.md`; después el manifiesto
> JSON y la última evidencia pública de T023-F. La migración de desarrollo a
> este Mac está autorizada; usá mi usuario y checkout locales reales, sin
> trasladar la ruta ni los privilegios del servidor. Confirmá rama, commit,
> cambios ajenos y estado de cualquier trabajo previo antes de repetirlo.
> Completá y verificá el entorno CPython 3.13 para esta plataforma sin copiar
> el venv Linux ni secretos/datos/modelos. Conservá los resultados Linux como
> históricos y no acredites hardware ni criterios físicos con pruebas Mac.
> Después continuá desde el próximo paso registrado para T023-G, sin reiniciar
> el roadmap ni crear un objetivo distinto. Mantené abiertos los criterios
> pendientes y separados permisos de captura, retiro, reactivación, voz,
> persona, cuenta y consentimiento. Producción y releases conservan su alcance
> autorizado por separado.

## Lista de cierre y reanudación

- [x] F terminada, revisada, commiteada y subida; registrar fuentes, métricas y
  evidencia final sin sustituir los intentos históricos.
- [x] Tracking y este handoff publicados/subidos; checkout y trabajos propios
  reconciliados, sin un full o Chromium pendientes que la sesión nueva duplique.
- [x] Pausa del mismo goal solicitada por el usuario al cerrar, sin declararlo completo; confirmar su estado al retomar.
- [ ] En Mac: usuario/ruta/arquitectura/CPython y dependencias comprobados;
  ninguna preparación de Mac se da por realizada sólo por este documento.
- [ ] Mismo roadmap y próxima unidad recuperados desde Git; registrar los
  primeros resultados del entorno nuevo antes de ampliar validación.
