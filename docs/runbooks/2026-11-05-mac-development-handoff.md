# Continuar el desarrollo de KZA en Mac

Estado del desarrollo al cerrar T023-G en el nuevo entorno Mac.
Conserva el traspaso Linux de T023-F y los resultados nativos posteriores
para retomar el mismo goal con su roadmap y evidencias.

## Punto de reanudación

La unidad **T024-A quedó verificada localmente en Mac**. El mismo goal sigue **activo e incompleto**. Usar la `.venv` nativa y no repetir tandas cerradas sin una razón nueva.

| Dato | Estado actual |
| --- | --- |
| Entorno | `gabriel`, `/Users/gabriel/Documents/Proyectos/kza`, macOS 26.6.1 arm64, CPython 3.13.18 |
| Código T024-A | `71971584430adbbaae425923d92f4a23a9ce7f25` |
| Baseline original A | 4 PASS + un RED genuino + un fallo del fixture; ambos preservados y clasificados |
| Integración A | 6 PASS; mismos seis IDs y presupuestos |
| Dirigida A | 589 PASS; XML 26.285 s |
| Full normal A | **8535 PASS + una xfail conocida**, 8536 IDs exactos; XML 673.755 s |
| Fuentes | 1028 archivos de fuentes/pruebas/útiles, incluidos 37 scripts; además 13 archivos protegidos intactos |
| Identidades | Todos los 8530 IDs de J más seis nuevos; sin renombres ni retiradas |
| Delegación | Hasta ocho agentes activos en total según plazas; pruebas interactivas de PC con `gpt-6.1-terra` |
| Próxima unidad | T024-B: restauración integrada e inventario de los diez participantes, con drivers reales |
| Tracking | Este commit; verificar SHA y publicación al retomar |

T022/T023/T024 siguen `in_progress` y los 21 criterios `not_evaluated`. T020/P1/T047, fases, presupuestos y permisos separados se conservan. Leer el [informe A](../research/2026-11-05-multistore-restore-rehearsal.json) y la [continuación B](../plans/2026-11-05-integrated-restore-continuation.json). Sin release ni cambios de producción.

## Referencia histórica del cierre Linux F

El usuario pidió terminar la unidad en curso, guardar el avance y retomar el
mismo goal desde una sesión nueva en Mac. El objetivo sigue en
`docs/plans/2026-11-02-autonomous-roadmap.md`; el estado operativo está en
`docs/plans/2026-11-02-autonomous-roadmap.json` y el historial en
`docs/runbooks/2026-11-02-autonomous-execution.md`.

| Dato | Estado al cerrar la sesión Linux |
| --- | --- |
| Rama | `recovery/production-2026-10-25` |
| Última entrega verificada antes de F | `be32294`, T023-E: 8294 PASS y una xfail conocida; 1010 archivos, incluidos 37 scripts, sin cambios durante esa full |
| Unidad entregada | T023-F: autoridad y retiro de captura/inferencia/reporte de `diagnose_wakeword` |
| Commit de código F | `38f3912d5272be408f9d27afb74a298a54d2b5ac`; comprobar su publicación con el historial de la rama y el registro de cierre |
| Full F y evidencia pública | 8341 PASS y una xfail conocida; 1013 archivos, incluidos 37 scripts, sin cambios; XML 851.448 s. Evidencia: [captura de diagnóstico](../research/2026-11-05-wakeword-diagnostics-capture-authority.json) |
| Commit de tracking y handoff Linux | `06cd9162e8516ff906ab092246e02edc499823ad`, histórico; no usar el último cambio Mac de este archivo como SHA del cierre Linux |
| Estado del goal al salir | Pausa solicitada por el usuario después de esta entrega, con el objetivo incompleto. Confirmar el estado del mismo goal al retomar; no crear otro objetivo ni marcarlo completo. |
| Procesos de desarrollo pendientes Linux | F terminó; sus handles son históricos y no se reutilizan en Mac |

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

Al cerrar el traspaso Linux todavía se desconocían esos datos. La verificación
inicial de la sesión Mac está registrada más abajo. Las herramientas Git y del
asistente se autentican por sus flujos propios; este documento no transporta
secretos.

## Comprobación y próxima unidad

Leer el [informe I](../research/2026-11-05-legacy-wakeword-capture-authority.json) y la [auditoría y propuesta J](../plans/2026-11-05-global-capture-followup.json). La fábrica real inicializa el control antes de los dos loops de producción, pero su OwnedInputStream compartido omite comprobación estricta de errores PortAudio y reintenta sobre un wrapper cuyo puntero puede haberse descartado. Es un hallazgo estático pendiente de reproducción aislada, sin ejecutar dispositivos o servicios reales. StreamingSTT permanece condicional sin llamador acreditado. La propuesta original I es histórica; no repetir su baseline ni sus pruebas por cambio de sesión.

## Verificación inicial en Mac — 2026-11-05

El usuario reactivó el mismo goal; la pausa del cierre Linux ya terminó.
El checkout de `gabriel` está en
`/Users/gabriel/Documents/Proyectos/kza`, sobre la rama registrada y el commit
de traspaso `06cd916`. El equipo ejecuta macOS 26.6.1 arm64; Serena apunta a ese
checkout y no tiene memorias de proyecto disponibles.

Se creó un `.venv` nuevo con CPython 3.13.18 nativo arm64. El venv trasladado de
Linux conserva su enlace roto a `/usr/bin/python3.13` y no se ejecutó. El
manifiesto completo de tests se instaló y `pip check` terminó sin errores;
Torch, SoundFile y Chroma son módulos reales. Los markers de PyTorch conservan
`2.12.0+cpu` fuera de Darwin y seleccionan `2.12.0` en Darwin, sin recortar el
resto de requisitos.

Pasaron los **31 tests iniciales de clientes** y el escenario de escritorio
C/D/E/F con Chrome.app, perfil temporal nativo y HTTP real. Sus 20 archivos
observados permanecieron intactos y se confirmaron los retornos del navegador
y del servidor. Los oráculos y cierres del fixture siguen iguales; Linux
conserva su binario `chromium` y su directorio snap. La UI usa assets locales
vendorizados: no hizo falta instalar paquetes npm.

La [evidencia de esta verificación](../research/2026-11-05-mac-development-environment.json)
distingue estos resultados del histórico Linux. Esta primera tanda no acredita
la suite completa Mac, hardware, interacción móvil, modelos ni producción. El
desarrollo de G y su baseline tienen evidencia separada; los 21 criterios
continúan pendientes.

## Captura MultiMic y zone_setup en Mac — G

El estado actual es G en Mac. El bootstrap está subido en
`26e549d8ecc4bc7b75029ff0f1e93518cbe661dd`; las tandas dirigidas y el browser G
ya terminaron. La segunda full Mac quedó verificada, con 8392 PASS y una xfail conocida. El código G está publicado en `43b44b933d8bff8a8a8bd2bdbd4f4d5555bfc743` y la portabilidad de fixtures en `da86bc5d84abbdd059f632d4934877b6e0953c45`.

Las APIs vivas `MultiMicCapture` y los comandos `test-mic`/`test-detection`
requieren un cliente nominal `zone_setup` y la autoridad administrativa
original. Los cuatro kinds nativos —calibración, grabación y diagnóstico de
wakewords, y zone_setup— comparten la barrera durable sin intercambiar
capacidades. El POST exacto de retiro admite capacidad opaca y Origin tras
logout; esa capacidad no permite captura ni reactivación.

El baseline conserva una caracterización aprobada y un RED genuino de arranque
parcial. La primera integración terminó con 20 PASS y 4 FAIL: dos oráculos del
DTO eran incorrectos y el reporte del CLI tenía un error real al adaptar
`publish(fn,*args)` a `print`. Se conservan resultados y causas; el segundo
bloque aprobó los mismos 24 casos. Otro RED confirmó que el retorno de un
`stop()` viejo podía borrar el estado de una captura nueva. La guarda de
identidad bajo lock lo corrigió; los 25 casos completos pasaron sin cambiar el
nuevo oráculo. El dueño nuevo conservaba sus handles: no se presenta esa carrera
como una fuga o ACK prematuro.

La dirigida Mac terminó con **653 PASS**, terminal 0, XML 131.673 s, sobre 1016
archivos incluidos 37 scripts sin cambios. La unión de IDs es exacta:
602 de F conservados, 5 backend, 9 cliente, 25 de integración G y 12 auth nuevos.
Los 58 del backend/cliente y los 25 de integración son grupos solapados, no
sumandos adicionales. El review final del dueño cerró estáticamente la carrera;
el runtime G25 conserva su recibo independiente.

La primera full Mac terminó con **8389 PASS, 3 FAIL y una xfail conocida**,
8393 IDs exactos, XML 740.423 s y guardas de 1016/37 sin cambios. No se acepta como
full válida. Dos casos del fixture de descriptores asumían `/proc/self/fd`;
el tercero falló en el cleanup Hermes al consultar `getpgid` después de una
carrera de salida. No son REDs del producto G. Se corrigieron sólo los dos
archivos de tests; las fuentes productivas G siguen iguales.

Los dos primeros replays de esos fixtures terminaron cada uno con 47 PASS y
1 FAIL: primero `killpg` devolvió EPERM en Mac; después el nuevo wait0 era un
oráculo innecesariamente inmediato. El patch final conserva exactamente el
único wait real original de 3 s y admite EPERM sólo después de ese retorno real;
no agrega otro plazo, reintento o skip. El tercero pasó los 48 casos, terminal 0,
XML 7.668 s. La sonda del observador Darwin `/dev/fd` mostró 4→5→4 y EBADF tras
cierre real; eso valida el observador, no sustituye los tests de producto.

Se conservaron en cuarentena privada 920 archivos ignorados de bytecode copiado
que guardaban `co_filename` Linux. La recompilación fresca usa la `.venv`
nativa; no cambiaron fuentes, flags, dependencias ni el venv Linux. La segunda
full terminó con 8392 PASS y una xfail conocida, sin fallos, errores ni omisiones ordinarias, en 583.548 s XML. Conserva los 8342 IDs de F y añade 51; sus 1016 archivos, incluidos 37 scripts, permanecieron intactos.

El browser G terminó con terminal 0 y ejercicio 11.696 s. React/Chrome y HTTP
reales comprobaron native 4→3→2→1 sin ACK y con resume deshabilitado; sólo el
retiro por capacidad de zone_setup alcanza 0/ACK, seguido por reactivación
explícita de la sesión original. Los cuatro participantes nativos de ese
fixture son registros HTTP del plano de control: el browser no ejecuta los CLI
ni prueba retiro de micrófonos o SDK físicos. La imagen de 390 px sigue siendo sólo
observación, sin aceptación móvil.

Los retornos de streams conocidos, callbacks admitidos, reportes y monitor
preceden el ACK. Cierres inciertos o constructores sin retorno mantienen la
barrera. Reanudar o recuperar permiso no arranca audio: hace falta un `start()`
nuevo. Los arrays ya cedidos a callback/cola no se borran como propios; las
lecturas históricas, copias, TTL y borrado/restauración conservan su alcance.

El [informe G](../research/2026-11-05-zone-setup-capture-authority.json)
registra el proof real de full, commits, publicación e historia de cada intento.
No hay aceptación de hardware, SDK privado, borrado físico, C04 ni de todos los
productores. T022/T023 siguen `in_progress`, T024 `planned` y los 21 criterios
`not_evaluated`; T020/P1/T047 y las autoridades separadas se conservan.

El usuario pidió cerrar G y continuar en otra sesión del mismo goal. La pausa
se solicita al finalizar la publicación. Confirmar su resultado real por API al retomar, junto al checkpoint Mac y Git. No crear otro objetivo ni marcar éste como completo.


La propuesta H queda en [continuación del diagnóstico de comandos](../plans/2026-11-05-manual-voice-command-continuation.json): sólo lectura y baseline sintético futuro, sin importar `.env` privada ni ejecutar HA.

## Diagnóstico manual bajo autoridades separadas — H

H usa kind fijo manual_voice_command y admin:manage original; la acción utiliza otra credencial home:control y el endpoint tipado existente. El cierre espera streams, iteradores, referencias propias de modelos, Chroma, reportes y monitor. Una respuesta HTTP200 sola no es éxito y un POST remoto sin recibo terminal conocido mantiene pendiente el retiro. No hay fallback a credenciales de captura ni lectura de .env para HA.

Se conservan todos los primeros resultados: watchdog de causa desconocida, dos errores de fixtures,4 RED genuinos y replay 33/1. El replay final 34 y la full validan el arreglo; esto no prueba audio/GPU/HA físicos ni borrado de SDK. La pausa mencionada en el registro histórico G quedó revocada por la reanudación posterior; el estado actual del goal es activo.

## CLI legacy de wakeword bajo captura autorizada — I

Ambos aliases conservan el wake word nexa. Las grabaciones y la prueba exigen administración original mediante el kind existente wakeword_training; no hay sexto kind ni nuevas rutas. El dueño conserva streams, detector, colas y operaciones hasta sus retornos reales, antes de ACK. Se retira systemctl implícito; train sigue offline y deploy sólo informa. Se conservan seis canales, copia del canal 0, RMS float32, filtro sólo positivo, WAV y semántica de score/threshold.

Se archivan el baseline 8 PASS, el primer 35/8 y el replay 43 PASS. Se restablece la propagación original de main; cinco fallos de los fixtures nuevos se corrigen con cambios explícitos, sin aumentar presupuestos ni rebajar barreras. La dirigida y full validan los bytes finales; esto no acredita hardware, modelo real, SDK privado, borrado ni criterios físicos.

## Recibos estrictos del input de producción — J

Cada handle devuelto y conocido recibe una llamada estricta a stop y close, incluso si stop falla. Un constructor admitido sin handle o un close fallido conserva UNKNOWN, el dueño y la prohibición de ACK/reapertura; un puntero nulo o Future terminal no prueba retiro. La no admisión comprobada antes del constructor mantiene la recuperación de dispositivos ausentes. Se conservan DSP, drenaje real de callbacks y el tratamiento previo de errores ordinarios de hooks.

Quedan archivados el baseline original, los fallos iniciales y sus correcciones, sin cambiar los 14 oráculos nuevos ni las cuatro regresiones originales. Dos antiguos oráculos de reintento se migran expresamente al contrato real del SDK. Las pruebas usan SDK/actores sintéticos; no acreditan hardware, memoria privada del SDK, borrado universal ni C04/C05.

## Restauración conjunta y recibos de capturas — T024-A

El journal reabre también capturas durante el arranque con lease. El baseline reprodujo un recibo completo con unlink real retenido tras restaurar datos compatibles. El segundo fallo era una lectura del fixture después del cierre correcto de su worker; se preserva la comparación de almacenamiento y se comprueban APIs antes del cierre. No se cambian seis identidades, barreras ni tiempos.

El ensayo comparte una autoridad para captura FLAC y conversación gestionada, restaura sólo datos mediante backup SQLite y filas de payload, y conserva tombstones, actor, operación, targets, fechas y vencimientos. La fuente ajena mantiene archivos y filas. Un caso controla el scheduler para demostrar lecturas denegadas con datos todavía presentes; los demás conservan retención real. La limpieza retenida se libera antes del cierre y se espera su retorno real.

Esto no acredita STT nuevo ni un agregado completo. Productores y los inventarios deliberadamente pendientes de diagnósticos, streams y voces siguen abiertos; no retirar esos targets ni simular sus recibos. El índice semántico ya tiene su propio recovery. El warning existente del logger por datetime.utcnow y la xfail conocida no se ocultan ni aceptan C05.

## Prompt para la sesión nueva

> Retomá el MISMO goal activo e incompleto de `docs/plans/2026-11-02-autonomous-roadmap.md`. Leé AGENTS.md, CLAUDE.md, este handoff, el informe T024-A y la continuación T024-B. Confirmá API del goal, checkpoint, usuario/ruta/arquitectura, rama, HEAD/origin y cambios ajenos. No crees otro objetivo ni marques éste completo.
> A quedó verificada: originales 4 PASS/un RED genuino/un fallo del fixture; finales 6/589 PASS y full 8535 PASS + una xfail conocida, 8536 IDs exactos. Usá la `.venv` nativa; no recrees dependencias ni repitas tandas sin razón nueva. Hasta ocho agentes totales según plazas; interacción de PC delegada a Terra.
> Comenzá B con los diez participantes y sus builders reales, misma autoridad y snapshots de datos compatibles. Conservá obligaciones unresolved, desconocidos, tombstones y datos ajenos; ningún callback ficticio ni eliminación de targets puede fabricar complete. T022/T023/T024 y los 21 criterios siguen abiertos; T020/P1/T047 y permisos separados se conservan. Producción/releases mantienen su alcance separado.

## Lista de cierre y reanudación

- [x] Linux y unidades G/H/I/J conservados con recibos propios.
- [x] T024-A con baseline original, correcciones y 6/589/full documentados.
- [x] Código, tracking y handoff guardados; verificar publicación y HEAD al retomar.
- [x] Mismo goal activo, sin nueva petición de pausa ni criterios físicos aceptados.
- [ ] Confirmar estado/Git y comenzar T024-B según continuación, sin repetir A.
