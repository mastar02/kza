# Continuar el desarrollo de KZA en Mac

Estado del desarrollo al cerrar T023-G en el nuevo entorno Mac.
Conserva el traspaso Linux de T023-F y los resultados nativos posteriores
para retomar el mismo goal con su roadmap y evidencias.

## Punto de reanudación

**T024-F1 sigue en curso: la base de admisión/cierre tiene 112 pruebas dirigidas aprobadas; el rollback completo aún no está demostrado.** T024-F0 permanece como última unidad completa verificada localmente. El mismo goal sigue **activo e incompleto**. Usar la `.venv` nativa y conservar los ensayos cerrados.

| Dato | Estado actual |
| --- | --- |
| Entorno | `gabriel`, `/Users/gabriel/Documents/Proyectos/kza`, macOS 26.6.1 arm64, CPython 3.13.18 |
| Implementación F0 | `3babff8aaa7dc8c5f87a5255e198a149ecfe718d`; builder y cuatro pruebas de regresión |
| Artefacto comparado | Revisión D `9b3170294eb69a53dd27b63949c09a63a2efed54`;436 archivos/378pyc |
| Tests F1 | `263e7458204c226aebfc64758831743bbd9f6119`; tres archivos nuevos, sin cambio de producto |
| Base de admisión F1 | `80b47f55716e980b93a7312b6732de0611e3e2e0`; 112 PASS/XML79.374 s, sólo prerrequisitos de fuente |
| Original F1 | **1 FAIL genuino**, XML14.430 s; tres hijos con wait real0; fallo final por autoridad vieja seleccionada |
| Autoridad F1 | Faltan tombstone S, borrado de perfil, revocación de credencial y reducción de permiso; U/manual preservados |
| RED F0 preservado | Variación por `sys.intern` reproducida antes del fix; parsear/retener sin internado conserva bytes |
| Green F0 | 4 PASS; XML24.557 s; caso original y worker intactos |
| Matriz F0 | Seis builds con bytes iguales; semillas0/1 y contexto limpio/internado |
| Dirigida F0 | 1234 PASS; XML 243.031 s |
| Última full, F0 | **8584 PASS + misma xfail conocida**,8585 IDs exactos; XML689.339 s; anterior a la regresión nueva F1 |
| Guardas del original F1 | 1046 fuentes,37 scripts y27 protegidos intactos; tres archivos nuevos con un caso respecto a F0 |
| Warnings F0 | 15 firmas pytest iguales a E; UserWarning Chroma del hijo conservado aparte |
| Alcance actual | Baseline original intacto; admisión, ownership y testigos de cierre en fuente; sin positivo gestionado compilado ni derivación R/F |
| Delegación | Hasta ocho agentes según plazas; PC interactiva con subagente `gpt-6.1-terra` |
| Próxima acción | F1: corregir admisión/derivación y agregar rollback positivo de código y datos, más recuperación hacia adelante |
| Tracking | Este commit; verificar SHA/publicación al retomar |

T021/T022/T023/T024 siguen `in_progress`;21 criterios `not_evaluated`. Tres guards, C04/C05, warnings y xfail siguen abiertos. T020/P1/T047, presupuestos y permisos separados se conservan. Leer [baseline F1](../research/2026-11-06-copied-slot-authority-baseline.json), [prerrequisito de admisión](../research/2026-11-06-runtime-admission-foundation.json) y [continuación de implementación](../plans/2026-11-06-slot-authority-reconciliation-continuation.json). Se conserva el RED genuino original; su harness aún debe adaptarse al lifetime gestionado. No se ejecutó ni aceptó una full con los cambios nuevos. Sin release ni cambios en producción.

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

## Restauración integrada e inventarios pendientes — T024-B

Los builders reales comparten PrivacyServices, journal, lease y el coordinador creado por el índice. Se restauran datos seleccionados sin retroceder autoridad, tombstones, ACK ni decisiones manuales. La fuente ajena conserva datos persistidos exactos. El contexto rehidratado recupera resumen/fuentes y no turnos volátiles, conforme a su contrato.

Dos gates retienen delete Chroma y unlink reales; cancelar un observador iniciado o liberar sólo unlink no completa el índice. CommandProcessor conserva el lease hasta retirar su STT nativo. Los primeros fallos eran supuestos del fixture sobre revisiones expiradas e historia volátil; se mantienen evidencia, identidades y tiempos.

Productores sigue sin driver y diagnósticos/streams/voces mantienen sus guards unresolved. El peer sintético y los WAV/transcripciones no acreditan navegador físico ni procedencia explícita donde el metadata permanece NULL. El warning del logger y la xfail conocida siguen visibles. C04/C05 y el goal permanecen abiertos.

## Productores por conversación y registro lazy — T024-C

ApplicationGraph registra PRODUCERS en el mismo coordinador/autoridad antes de admitir trabajo. Revalida roles, alias, modelos, borrowers y delegates; un reemplazo observado invalida el sello aunque vuelva el objeto anterior. Target v1 estable; lease nueva reabre la prueba y la misma lease no la reinicia. Claim y tombstone confirmado se verifican antes/después, con observación final sin await antes del resultado.

S se retira mientras U sigue dentro de STT real, sin pausa global ni renovar sus fuentes. Wake se registra antes de load bajo un lock breve de metadata; primera adopción lazy acreditada y reemplazo posterior invalidado. Hermes conserva cuerpo y alias al reubicar su clase nominal. Ambient STT/speaker comparten ledger y registry comprobados.

El primer fallo intentaba crear un estado prohibido por FK. El test comprueba rechazo y petición válida, conservando IDs/schema/tiempos. Tras revisión, se corrigió autoridad anidada ambient y se verificó con cuatro casos reales. El child UNKNOWN mantiene pendiente el recibo y la lease; su salida sólo contiene el proceso, sin acreditar retirada del SDK. Tres guards, C04/C05, warnings y xfail conocida siguen abiertos.

## Restauración por el grafo público — T024-D

Dos generaciones de `build_application`, diez drivers y nueve familias de payload S/U. El productor S espera su retorno real mientras U continúa. La nueva lease reabre recibos; inventario nuevo, claim viejo rechazado y S revocado bloqueado antes del backend. VoiceLedger oculta S con BLOB presente antes de construir voces; vector/unlink reales retienen sus pasos hasta retornar.

Se conserva toda fila vigente, incluidas tablas mixtas, perfil confirmado, link/unlink por CAS, deletion fence, grupo, hecho confirmado, respuestas de sugerencias, persona y credencial HOME sin permisos de revisión. El testigo G1 conserva linaje de índice sin restaurar fact_sources, autoridad ni ACK. El preflight rechaza conflictos deterministas antes de escribir payload; no acredita atomicidad ante un crash entre stores. Tres guards y criterios siguen abiertos.

## Grafo desde bytecode en procesos nuevos — T024-E

Un archivo compilado exacto por ensayo, dos intérpretes nuevos `-I -S -B` y una sola raíz física de estado. Imports src restringidos al manifest .pyc;18 helpers y entry separados, sin conftest/src fuente/.env. JSON tipado/blobs por hash conserva el testigo D sin transportar objetos vivos. Ambos grafos públicos ejercitan diez drivers, nueve familias, fuentes/manuales/permisos vigentes y retornos nativos reales; tres guards impiden aggregate complete.

Original1 PASS, dirigida1230 PASS y full8580 PASS+misma xfail/8581 IDs. Warning Chroma del embedding sintético conservado. Entre builds de la misma revisión hay5pyc distintos con headers/campos de código canónicos iguales; causa pendiente, sin prometer igualdad binaria. La construcción reproducible sigue siendo requisito y F0 la investiga antes del rollback entre versiones. La corrección de un NameError del generador privado de recibos no cambió fuentes ni repitió pruebas.

## Compilador independiente del proceso llamador — T024-F0

El contracaso original queda preservado: `sys.intern` altera los bytes en dos módulos, mientras parsear y retener las mismas cadenas no los altera. El builder ahora usa un compilador nuevo con semilla fija, `py_compile` estándar y validación de ABI, inventario y headers. Conserva errores y warnings nativos y la publicación sin sobrescritura. La igualdad completa del archivo sigue siendo el oráculo.

Primer green4 PASS, matriz original de seis builds reproducibles, dirigida1234 PASS y full8584 PASS+misma xfail/8585 IDs exactos. Los cuatro archivos de cada regresión y el artefacto del grafo compilado coinciden también entre tandas. Cabeceras y campos comparados de los378pyc coinciden con la salida anterior, con dos diferencias binarias; esa comparación estructural limitada no reemplaza el oráculo ni prueba equivalencia universal. F1 conserva la obligación de rollback positivo entre versiones con autoridad y datos vigentes, más recuperación hacia adelante.

## Copia de slot y autoridad vigente — T024-F1, baseline

El original ejecutó `build_runtime`, `prepare_slot`, `activate_slot`, dos grafos públicos desde bytecode y un inspector SQL/SDK sin grafo ni lectores. Los tres hijos terminaron con wait real0; 277/277/269 módulos src provinieron del manifest. Se conservaron U, manuales, TTL y retornos nativos, con guards diagnostics1/streams2/voices2 pendientes. El fallo final ocurre porque se admite seleccionar el estado viejo sin cuatro decisiones actuales: tombstone S, perfil borrado, credencial revocada y permiso reducido. Consentimiento deny se mantuvo como input; no se probó una transición de revocación de consentimiento ni una fuga servida por un lector viejo.

La limpieza por rutas históricas eliminó derivados conocidos de S sin alterar U/manual; eso no se clasifica automáticamente como bug. El SDK viejo sólo se consultó para U. El archivo privado de 982 entradas conserva datos y metadatos del ensayo como evidencia, no como runtime relocalizado ni testigo válido para otra raíz. Original1 FAIL/XML14.430 s, sin cambio de producto y sin xfail nuevo; revisión independiente confirma el RED. La última full aceptada sigue siendo F0, anterior a este test requerido que falla.

Conservar este gate de rechazo de selección vieja separado del positivo futuro R/forward. R debe exponer un punto ordinario anterior real con autoridad/manual/permisos/consentimiento latest; luego F debe recuperar datos hacia adelante conservando una revisión protegida nueva hecha en R. No basta alias al estado latest, rechazo de código incompatible ni un cambio de puntero. El par A719/C598 inspeccionado no acredita compatibilidad completa por revalidación de recibos de productores; hace falta un par semántico compatible real.

## Admisión y cierre de slots — T024-F1, prerrequisito

El commit `80b47f55716e980b93a7312b6732de0611e3e2e0` agrega head monotónico, bindings, lease durante todo el hijo, delegaciones históricas explícitas y admisión antes de DI/SQLite. La retirada intenta siempre cleanup y wait aunque falle la escritura de UNKNOWN. La configuración efectiva queda ligada al YAML canónico; la retirada de consentimiento sólo se encola para la generación siguiente, sin afirmar revocación de políticas cacheadas en vivo.

La dirigida terminó con **112 PASS**, XML79.374 s, wait real del PID50473 y fuentes1054/scripts37/protegidos29 intactos. AST/Ruff pasan. El intento previo se conserva: 105 PASS/5 FAIL por exigir Chroma abierto antes del warmup; se corrigió observando su constructor real. Los casos distinguen owner nunca abierto de cliente adquirido y cerrado mediante el SDK público. Al registrar ese prerrequisito, los tres archivos del original263e seguían idénticos.

Ese reporte verifica prerrequisitos en fuente. Sus checks compilados legacy usaron HEAD9d76555, anterior a estos módulos; todavía no incluían el recorrido gestionado NEW/OLD, el cierre explícito de HA/Auth/capture auxiliares, el consentimiento siguiente ni R/F material. Si AudioLoop pierde una identidad lazy antes del stop-seam, se rechaza el cierre. El helper de reconciliación era parcial en esa publicación; el hito posterior se describe a continuación. Ver [alcance y evidencia](../research/2026-11-06-runtime-admission-foundation.json).

## Derivación y recorrido material — T024-F1, hito local

Los commits `4f88ac9`, `36cc99c`, `c30b12c`, `8a05398`, `3c45902` y `d574075` implementan el harness gestionado, snapshots cerrados, selección/materialización de nueve familias y journal local con historia calificada. El oracle original se conserva; el harness adaptado rechaza la autoridad stale. No fusionar autoridad con SQL ni renovar fuentes o vencimientos.

La suite de componentes terminó con **90 PASS**, XML1.910 s, wait real 32760/722db9. El recorrido compilado **O/N/R/F pasó**, XML43.690 s, wait real 37103/04e71d: cuatro estados físicos, código 9b en O/R y d574 en N/F, bridge80b para OLD, cuatro hijos con wait0 y 277/279/277/279 orígenes bytecode. R conserva decisiones latest y F conserva una revocación adicional hecha en R; ambos consumen el consentimiento `False` de la generación siguiente. Fuentes1064/scripts37/protegidos30 intactos. Se conservan los fallos previos: retención numérica, captura repetida del fixture y export manual omitido; sus correcciones no rebajaron los controles.

**F1 sigue abierto.** La proyección de admisión V1 y el clasificador histórico de cutproof todavía no abarcan por sí solos el cierre export/padres manuales. No cambiar V1 silenciosamente: el bridge80b recaptura el binding. Faltan cobertura negativa y checks generales/guardias; última fullF0 permanece8584PASS+xfail conocida, anterior aF1. `native_gates_returned=null` en O/R/F no cuenta como ACK universal. T021–T024 y los 21 criterios no cambian. Ver [reporte con alcance y evidencia](../research/2026-11-06-state-derivation-progress.json).

## Prompt para la sesión nueva

> Retomá el MISMO goal activo e incompleto de `docs/plans/2026-11-02-autonomous-roadmap.md`. Leé AGENTS.md, CLAUDE.md, este handoff y el reporte `docs/research/2026-11-06-state-derivation-progress.json`. Confirmá goal/checkpoint, usuario/ruta/arquitectura, rama, HEAD/origin y cambios ajenos. No crees otro goal ni marques éste completo.
> Hito F1: componentes 90 PASS y primer material O/N/R/F 1 PASS/XML43.690 en d574075, cuatro procesos wait0/cerrados; bridge80b para OLD9b y NEWd574 real. Sus controles no equivalen a aceptación universal. No repetir ensayos cerrados sin causa nueva. Máximo ocho agentes según plazas; navegador/escritorio de PC siempre delegado a un subagente Terra.
> Próximo: completar clasificador histórico de exports/padres manuales y su regresión negativa, diseñar garantía de admisión compatible con V1/bridge80b, luego full/guardias y revisión universal de cierres. Conservá original RED, oracle, proveniencia/TTL y autoridad latest. No reetiquetar testigos ni dar por entregado sólo rechazo, alias latest o un positivo aislado. T021/T022/T023/T024 y los 21 criterios siguen abiertos. Sin release/producción/campo.

## Lista de cierre y reanudación

- [x] Linux y unidades G/H/I/J conservados con recibos propios.
- [x] T024-A/B/C/D conservadas y T024-E con original1, dirigida1230, full y limitaciones documentados.
- [x] T024-F0: RED original, green4, matriz6, dirigida/full y revisión independiente conservados.
- [x] F1 original: RED genuino, tres cierres/waits0, archivo privado y revisión independiente conservados; sin fix de producto.
- [x] Código, tracking y handoff guardados; verificar publicación y HEAD al retomar.
- [x] Mismo goal activo, sin nueva petición de pausa ni criterios físicos aceptados.
- [ ] Corregir F1 y verificar rollback positivo de código/datos más recuperación; la regresión nueva sigue fallando.
