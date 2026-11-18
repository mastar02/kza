# Continuar el desarrollo de KZA en Mac

Estado del desarrollo al cerrar T023-G en el nuevo entorno Mac.
Conserva el traspaso Linux de T023-F y los resultados nativos posteriores
para retomar el mismo goal con su roadmap y evidencias.

## Punto de reanudación

**Sesión cerrada y goal pausado por pedido del usuario.** Producto `c0e82df` publicado; 91 pruebas dirigidas y 2 nativas aceptadas dentro de su alcance. La full de 9382 casos quedó incompleta: 5620 PASS + 6 FAIL, corte por deadline de 1500 s, terminal `19778/77d9b6` exit 1. Exportación `81344/a50ea3` consumida; no quedan ejecuciones pendientes. La última full aceptada sigue siendo el hito histórico V1 de `66951c5`.

Leer [2026-11-07-session-handoff.md](2026-11-07-session-handoff.md) para evidencia, seis fallas, límites y prompt de goal nuevo. Primero reparar la liberación de referencias privadas y revisar el presupuesto de la suite; cuentas/V3 sigue sólo en diseño. Siete familias instrumentadas, dos registros primarios cerrados, cinco parciales y 30/32 obligaciones pendientes; sin ACK, release ni cambios de producción.

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

Registro histórico hasta `d574075`; el avance vigente figura en la sección siguiente y en el reporte actual.

Los commits `4f88ac9`, `36cc99c`, `c30b12c`, `8a05398`, `3c45902` y `d574075` implementan el harness gestionado, snapshots cerrados, selección/materialización de nueve familias y journal local con historia calificada. El oracle original se conserva; el harness adaptado rechaza la autoridad stale. No fusionar autoridad con SQL ni renovar fuentes o vencimientos.

La suite de componentes terminó con **90 PASS**, XML1.910 s, wait real 32760/722db9. El recorrido compilado **O/N/R/F pasó**, XML43.690 s, wait real 37103/04e71d: cuatro estados físicos, código 9b en O/R y d574 en N/F, bridge80b para OLD, cuatro hijos con wait0 y 277/279/277/279 orígenes bytecode. R conserva decisiones latest y F conserva una revocación adicional hecha en R; ambos consumen el consentimiento `False` de la generación siguiente. Fuentes1064/scripts37/protegidos30 intactos. Se conservan los fallos previos: retención numérica, captura repetida del fixture y export manual omitido; sus correcciones no rebajaron los controles.

**F1 sigue abierto.** La proyección de admisión V1 y el clasificador histórico de cutproof todavía no abarcan por sí solos el cierre export/padres manuales. No cambiar V1 silenciosamente: el bridge80b recaptura el binding. Faltan cobertura negativa y checks generales/guardias; última fullF0 permanece8584PASS+xfail conocida, anterior aF1. `native_gates_returned=null` en O/R/F no cuenta como ACK universal. T021–T024 y los 21 criterios no cambian. Ver [reporte con alcance y evidencia](../research/2026-11-06-state-derivation-progress.json).

## Autoridad manual y construcción — T024-F1, avance local

Registro histórico hasta `3fc5ac6`; consultar el avance vigente del catálogo en la sección siguiente.

Los commits posteriores a `d574075` agregan el clasificador histórico de exports manuales, componentes tipados sin copiar contenido, sellos externos de autoridad manual y testigos de cierre de HA/Auth/captura con propietarios reales. Los contratos V1 no cambian; un HEAD anterior sin sello falla cerrado y todavía no tiene migrador.

La barrera de construcción en `3fc5ac6` mantiene excluido el mantenimiento histórico de diagnósticos hasta el retorno real de la validación nativa. La cancelación espera ese retorno y los escritores ajenos siguen rechazados. Pasaron 21 pruebas de construcción y dos regresiones compiladas con la aplicación anterior y el puente actualizado. O/N/R/F registra cuatro cierres y wait0 reales; esto acredita el escenario declarado.

Las full de f550 y df88 fallaron y se conservan como tales. La colección nueva retiene sus 8823 IDs y agrega diez, sin quitar casos. La full nueva terminó con8832 PASS y la misma xfail conocida,8833 IDs exactos; las15 firmas de warnings se mantienen. También falta un catálogo durable antes del primer byte para todas las copias y sus obligaciones de limpieza/recuperación. El case físico temporal de un ensayo d853 anterior fue retirado por pytest; sólo se conservan sus recibos y logs. Los cases nuevos de3fc se exportan con su procedencia, sin reconstruir testigos anteriores. Ver el [reporte actual](../research/2026-11-06-manual-authority-progress.json).

## Catálogo durable de copias — T024-F1, base local

Este apartado conserva el hito histórico `2cdbe51`; el avance posterior se registra a continuación.

`2cdbe51` agrega el catálogo externo antes del primer byte, identidad física, eventos inmutables y CAS. Los snapshots usados por el materializador requieren su captura externa prepared+finished; un DTO o manifest local no concede permiso. La incertidumbre del SDK es persistente y la capacidad se libera con `finish()` después de la última publicación.

El corte local pasó343 componentes, dos regresiones compiladas y8890 casos generales con la única xfail conocida. Un proceso nuevo cargó `copy_catalog.pyc`, leyó seis copias finalizadas y comprobó seis retornos SDK. Otro probe dentro de ese lector rechazó una factory que devolvía un cliente Chroma real prestado, sin invocarla ni confirmar limpieza. Casos físicos, metadatos, warnings y testigos se conservan con hashes; los archivos temporales antiguos ausentes no se reconstruyen.

F1 sigue abierto. El catálogo mantiene `cleanup=blocked, owner=None`; faltan owner y recibos de limpieza efectiva, más evidencia de fallos reales SDK bajo la capacidad nueva e inventario de copias retrospectivas/forenses. La retirada de S debe preservar exactamente U/manual y tratar la invalidez de snapshots históricos sin reescribir IDs ni pins. Ver el [reporte y requisitos pendientes](../research/2026-11-06-copy-lifecycle-progress.json).

## Limpieza selectiva de copias — T024-F1, avance local

`ae40a75` verificó fallos de persistencia en `handle_returned` y `close_entered` con constructor y cierre públicos reales:44 unidades PASS y2 compiladas PASS/XML140.734 s. Un lector nuevo recuperó los mismos IDs como UNKNOWN, sin permiso ni recibo de limpieza. Son fallos de markers, no fallos reales del constructor o cierre del SDK.

`94b83f1` incorpora `CollectorCopyCleanup`, claims CAS separados del catálogo y fences inmutables anteriores a la mutación. Bloquea admisión, referencias de snapshot y reutilización de la ruta invalidada, incluso con otro inode. Conserva los formatos V1 y los IDs/pins históricos. La cancelación del observador no libera la tarea, lease ni recursos nativos antes del retorno y cierre reales.

El escenario integrado registra ocho copias y limpia la representación collector de dos copias derivadas inactivas, en ejecución normal y cancelada. Retira S y exports dependientes sin retirar copadres U; conserva filas, archivos manuales y autoridad F. Un lector compilado nuevo comprueba ambos recibos y rechaza ambas admisiones. No equivale a borrar índices semánticos, snapshots, BLOBs, copias forenses ni otras representaciones.

Pasaron388 componentes,1 integrada y la full8936 con8935 PASS y la única xfail conocida. Las esperas reales fueron consumidas; la auditoría full pasó25 checks, sin firmas nuevas de warnings. Evidencia local ignorada por Git en `data/autonomous-roadmap-20261102/t024/f1/selective-cleanup-full/`: `acceptance-proof.json` SHA `15ef618c1f9c7ab7dd8f1f1fc3e795c75973d61ee44c974e1d37656a26bf65dd`, auditoría SHA `bd982cf1bbe4c909061628b4e9d6d7edfb9e29ac038754269ef66f25817e6110` y cinco casos físicos con manifest SHA `23f99afa0c32c1bc1a924486a2d80893e874e3e25bfd3fda17d0559e6468ddf2`. Los exports son evidencia histórica, sin permiso de admisión o limpieza. El código y los documentos se suben; estos datos privados permanecen locales.

## Constructor SDK sin handle — T024-F1, avance local

`180ce95ff0b6d141da1cdf81633ae9a791c08988` conserva los oráculos anteriores y agrega dos procesos compilados al escenario F cerrado. El constructor nominal consulta un tenant válido pero inexistente y propaga el mismo `NotFoundError`; el trace observa cero handles y cero llamadas a `Client.close`. Su `return None` corresponde al desenrollado de la excepción. No se reemplazaron factory, cierre ni flags internos.

El lector nuevo encuentra el mismo ID, SHA del registro e inventario físico parcial, con UNKNOWN/unfinished/blocked, sin capacidad o receipt. F/U/manual permanecen exactos. Los dos procesos retornaron wait0; eso no confirma retiro interno del SDK. Pasaron44 pruebas de protocolo/XML0.395 s y2 compiladas/XML138.932 s, con guardas1074/37/34 intactas durante la corrida compilada. Las44 unidades precedieron el último cambio del trace de integración; el código de producto y las unidades no cambiaron. No se repitió la full por esta extensión de prueba.

Evidencia local: `data/autonomous-roadmap-20261102/t024/f1/sdk-constructor/acceptance-proof.json`, SHA `8e75a037e6e2fef244c10bd03ecfe09b6db01de2c80aee962f9fdcdf77b50941`. Dos casos físicos conservados con manifest SHA `9c1a543320ca76a1e27b05b6e5a0c131e4d99ccbd1b62f1daa3f38a5d3f30973`. El fallo real de cierre sigue abierto: la revisión local no demostró una vía pública soportada para provocarlo sin sustituir métodos o forzar estado. Continuar con el inventario retrospectivo bajo autoridad vigente, conservando esa obligación pendiente.

## Observaciones retrospectivas y cota de registros — T024-F1

`dd637f7` añade `copy_observation`: validación de controlador/FD/HEAD sellado vigente, intent durable antes del escaneo y dos eventos inmutables con CAS en `admission/copy-observations/v1`. Los bytes ya existían: `before_first_byte=False`, `capability=False`, lineage incompleto y limpieza bloqueada. El registro no crea payload, ni prepara, migra o borra la copia. Conserva el ID del SDK fallido y todos los pins originales. La recuperación de metadatos requiere lease real y HEAD cerrado; no abre stores ni afirma sellado del payload.

Las rutas observadas y sus ancestros/descendientes quedan fuera de admisión, tanto para state como runtime, tickets con `check_state=False`, referencias de snapshot y destinos nuevos. Un cambio de inode no elude la denegación histórica. Un intent incompleto o un evento no publicado conserva UNKNOWN/prefijo validado, sin reparación automática ni ACK. Los formatos V1 y la barrera previa de limpieza se conservan.

El caso compilado cubre tres raíces regulares existentes y un lector nuevo, con F/U/manual y catálogo original exactos. No equivale a un censo de copias ni a owners de borrado. La full dd637f7 pasó8972 IDs, con8971 PASS y la única xfail conocida; auditoría sin incidencias y15 firmas de warnings anteriores. Evidencia local: `data/autonomous-roadmap-20261102/t024/f1/copy-observation/full/acceptance-proof.json`, SHA `02de81ad907a0078d915c538614d47ba30c75e53d9c3b1265ee4c611ab3b3035`. Seis casos físicos preservados, manifest SHA `3ea906da363d2128914fe60d77b508e8e9c598b8a3e21ee23ff2391f606246f3`.

Después se corrigió una diferencia productor/lector: `a2c6f4c` rechaza intent/evento canónico+LF mayor que16MiB antes de publicar. El intent de75.000 descriptores ocupa18.450.001 bytes sólo en el array y se rechazó antes del namespace. Un test de protocolo con seam2KiB conserva rev1 sin evento2/recibo; las tres observaciones normales siguen funcionando. Evidencia36+1: `copy-observation/bounds/acceptance-proof.json`, SHA `a7275be3f841efd922bc7c1a54059b56730a4c3415c1f4dc63721ade3b978649`. Estas verificaciones posteriores no se presentan como una full de a2c6f4c.

El análisis del caso nativo dd637f7 encontró14 BLOBs y43 referencias tipadas:7 manuales y36 de otra conversación preservada, con dos digests compartidos; ninguna acredita un binario de S retirado. Primero agregar un fixture opt-in con adquisición binaria pública S/U/manual/shared y probar los joins tipados. No aceptar como retirada un owner que no tenga ningún BLOB S elegible. Mantener abiertos SDK close real, cobertura/owners agregados, migración legacy, guards y los21 criterios.

## Lineage binario S/U/manual/shared — T024-F1, baseline local

`f487f40e891fd3ec5e0e092f45cee49cd85d8201` agrega únicamente pruebas y un fixture opt-in: tres correcciones STT públicas con audio real sintético, S privado, S compartido y U compartido marcado usado. Conserva fuentes, IDs, fechas, TTL y los oráculos originales. El analizador stdlib interpreta el wire sin crear DTOs de aplicación; verifica pins externos, los cuatro checkpoints originales, cada join de fuente/fila/archivo y la unión completa del directorio. Bloquea DTOs desconocidos, namespaces ajenos, fuentes SQL mal formadas y localizadores conflictivos antes de clasificar por digest.

Pasaron 68 unitarias, una integración opt-in y el recorrido original compilado, con 1081 fuentes/37 scripts/34 protegidos intactos en cada corrida. El caso nuevo contiene 16 BLOBs/49 referencias: 1 S-only,3 shared,12 retained,0 blocked; las 7 referencias manuales se conservan. El lector 37894 terminó con wait0 y el mismo informe sin importar aplicación/SDK. F y su inventario permanecen iguales; el registro retrospectivo del exchange no concede capacidad. No se borró el BLOB S. La full sigue siendo dd637f7; no se repitió para esta unidad de pruebas.

Evidencia local: `data/autonomous-roadmap-20261102/t024/f1/binary-lineage/acceptance-proof.json`, SHA `4af869e3cb0befdbaeca8b8973744eb28b9f5b4fd24973d8e59a812a8117468d`. Exportación de los dos casos sintéticos con manifest SHA `be59a807cfba26ba1ffd796c9892c21dce71ea2e59ed8fc7f6b06cd48cbf0ce4`; 5517 archivos/24 symlinks, originales y copias iguales. Los datos crudos están ignorados y no se suben a Git. No quedan procesos propios pendientes.

Siguiente: implementar retirada real del S-only bajo autoridad vigente, preservando shared/U/manual y JSON históricos con sus pins originales. El análisis de lineage y las observaciones no son permisos de escritura ni recibos de limpieza. Cobertura global, SDK close real, legacy, campo, F1/T024 y los 21 criterios siguen abiertos. Última unidad completa: T024-F0. Sin release ni cambios en producción.

## Pausa solicitada después de componentes del owner

Este apartado conserva el estado histórico de la pausa; la reanudación verificada figura a continuación.

La prueba inicial tuvo533 PASS+1 FAIL por un seam de test que parcheaba un alias distinto de stage; también se corrigió un ID de parámetro dict. `e1bcdb6` modifica sólo el test: el fallo se inyecta en el companion real después del evento durable y antes de HEAD. El segundo intento pasó534/534 exactos; primera evidencia fallida e inputs preservados. Las revisiones estáticas de parser/owner/integración no encontraron blockers.

Prueba local: `data/autonomous-roadmap-20261102/t024/f1/checkpoint-retirement/component-acceptance-proof.json`, SHA `c21a2d526c8553ce73d50a4f608e3c0fad52de7106713f0c2f73b41ee31c4055`. La exportación conserva8040 archivos/299 symlinks/327653762 bytes, manifest SHA `359686b1cf44143ea34ccd3530da83e517d9172047c51b715d550ca2a79f5c4e`. Código/documentación se suben a Git; datos sintéticos crudos y checkpoint permanecen locales e ignorados. No se afirma retirada compilada, ACK nuevo ni nueva full. Última unidad completa:T024-F0; última full:dd637f7.

## Reanudación verificada del owner — 2026-11-06

Se consumieron las esperas reales de native (`47477/7b16b9`), baseline (`95923/59a726`), default (`51938/44d8a5`), colección (`2310/473933`) y full (`33902/5ca693`), todas con exit 0, sobre `15713004e88c4551c608c294696bd7123c77b1fa`. No se repitieron los534 componentes. Native/baseline/default pasaron1+1+1; colección9085 exactos y full9084 PASS+la misma xfail regex conocida, XML1377.596s/resumen1377.89s. Guardas1084fuentes/37scripts/34protegidos intactas;65933 warnings,15 firmas idénticas a las previas, sin nuevas ni supresión.

El owner de producción se ejecutó en el proceso padre bajo autoridad del F compilado; el hijo nuevo cargó el parser `checkpoint_lineage.pyc` fijado al release. Se verificó un unlink real S-only,15 BLOBs conservados, cuatro checkpoints históricos intactos, cancelación con drenaje y recibo idempotente limitado a `checkpoint_blobs_s_only_v1`. No se acredita que el owner haya corrido dentro del hijo compilado ni un ACK agregado.

Evidencia local ignorada: `data/autonomous-roadmap-20261102/t024/f1/checkpoint-retirement/directed-acceptance-proof.json` (38checks) y `full/acceptance-proof.json`, SHA `ea1e34ae479bd3dc33d22ae0ab02ee5f58be6644efeb92845034aea46790cbb3` (audit31checks). La exportación conserva37483 archivos/2512 symlinks/1 FIFO/1888537903bytes, con originales y copias exactos. El exportador inicial rechazó el FIFO inerte de una prueba negativa antes de copiar; se conservó ese fallo y una variante limitada a su path/tipo/permisos lo reprodujo sin abrirlo ni leerlo. No fue fallo de la suite ni motivó repetirla.

La unidad seleccionada en aquel hito fue registrar productores/destinos antes de abrir/escribir en el grafo real, reutilizando autoridad, permisos y owners existentes. Diseño privado `checkpoint-retirement/next-census-design-v2.json` y revisión `checkpoint-retirement/v2-review.json`: journal acotado, primera integración listas/recordatorios y matriz completa de32 obligaciones. Resolver las cuatro precisiones de revisión al implementar; dos stores no equivalen a censo global. No repetir estas pruebas sin cambio o fallo nuevo.

F1/T024 siguen incompletos: censo y owners globales, fallo genuino de cierre SDK, migración legacy, guards diagnostics1/streams2/voices2 y aceptación física pendientes. Última unidad completa:T024-F0; T021–T024 in_progress y21 criterios not_evaluated. Goal activo; voz/persona separado de permisos; sin release ni cambios en producción.

## Registro de destinos listas/recordatorios — hito histórico V1

Registro durable de destinos de listas/recordatorios implementado en `ef8bc6d` y verificado localmente. Los 165 componentes pasaron (XML 19.571 s; 26 comprobaciones), con correspondencia exacta del candidato probado al commit. El primer intento 164 PASS + 1 FAIL por una llamada incorrecta de la prueba se conserva; sólo esa llamada se corrigió. El recorrido compilado pasó 1 prueba (XML 126.93 s; 44 comprobaciones), con cuatro registros N/F y lector nuevo fijado al release. Esas pruebas conservan su alcance `ef8bc6d`; no se afirma que se hayan repetido en `66951c5`.

La corrección `66951c5` cambia sólo `tests/integration/test_context_privacy.py`: el caso de saturación cancela y espera la tarea real de mantenimiento, conserva sus owners y drena el trabajo nativo antes de ocupar los ocho slots. Mantiene ocho workers reales, identidad de caché, comprobaciones de capacidad y timeouts; los demás casos conservan mantenimiento normal. La correspondencia exacta acredita producto y scripts sin cambios. Pasaron 26 pruebas dirigidas (XML 2.37 s/resumen 2.38 s), terminal `78590/78ec04` exit 0 y auditoría de 67 comprobaciones. Son un subconjunto de la full; no agregan IDs.

La full aceptada es el intento 03 en `66951c5`: 9125 PASS + la misma xfail regex conocida, 9126 IDs exactos (9085 previos + 41 nuevos), XML 1302.131 s/resumen 1302.33 s, terminal real `36106/10be8d` exit 0. Se respetó el límite fijo de 1500 s; 1086 fuentes / 37 scripts / 34 protegidos intactos. Hubo 66527 warnings y 15 firmas previas idénticas, sin firmas nuevas, ausentes ni supresión.

Los dos intentos full anteriores en `ef8bc6d` permanecen rechazados y conservados. El 01 (`21117/40ff09`, runner 1/pytest -2) excedió el límite durante la limpieza final, aunque imprimió 9125 PASS + xfail. El 02 (`24131/820428`, runner 1/pytest 1) terminó con 9124 PASS + 1 FAIL + xfail: falló la preparación de saturación antes de llegar a la aserción de identidad de caché. Un único sondeo controlado sobre código sin cambios pasó 14 comprobaciones y demostró una intercalación posible con mantenimiento; no reconstruye la cronología histórica de ese fallo. Las exportaciones exactas y los diagnósticos de ambos intentos se conservan.

Las bases N/F ya existían como estado copiado: el registro ocurrió antes de abrirlas este owner, no antes del primer byte histórico. El journal no concede autoridad, adopción retroactiva ni cleanup ACK; los owners originales conservan apertura, trabajo y cierre reales. La matriz tiene 32 categorías u obligaciones: dos familias instrumentadas y 30 categorías sin cubrir, además de huecos anidados/lazy/offline/externos/desconocidos. No prueba un total de 32 writers individuales.

Evidencia: [reporte de registro](../research/2026-11-06-output-registration-progress.json) y `data/autonomous-roadmap-20261102/t024/f1/output-registration/context-capacity-fix/acceptance-proof.json`, SHA `5a36a61f00e01ffb9472c78a745f08fb40e1e0631d390fefdaae14cd1f831e87`. La auditoría full tiene 49 comprobaciones. Su exportación conserva 37831 archivos, 2533 enlaces y 1 FIFO inerte, con originales/copias exactos según el verificador consumido. El retiro S-only previo conserva su propia evidencia histórica: owner en el padre bajo autoridad del F compilado; hijo nuevo con parser `.pyc` fijado al release. No se atribuye ejecución del owner al hijo ni un ACK agregado.

Siguiente bloque: continuar el censo global. El segundo parche privado v2 para privacidad, sesiones/dashboard, rutinas, eventos y sugerencias tiene revisión estática independiente sin bloqueos de código abiertos en su alcance, pero sigue sin aplicar ni ejecutar y sin aceptación runtime. Sus pins de parche, intención y revisión figuran en el reporte. Revalidar los originales antes de aplicarlo, recopilar IDs exactos y ejecutar las validaciones dirigidas, compiladas y full pendientes, conservando los huecos explícitos de cobertura. T024-F1 y T024 siguen incompletos; T024-F0 es la última unidad completa. T021–T024 siguen in_progress y los 21 criterios not_evaluated. Cierre SDK genuino, migración legacy, guards diagnostics1/streams2/voices2 y aceptación física pendientes. Voz/persona separado de permisos; sin release ni cambios de producción.

## Ciclos de retención de la full fallida — T024-F1, reparación local

Continuación en Claude Code (2026-11-09) sobre la full fallida `c0e82df` (5620 PASS + 6 FAIL, incompleta al límite de 1500 s). Las seis fallas reproducen de forma determinista en aislamiento (6 FAIL/74 PASS en 16 s), sin depender del orden ni del presupuesto. Una bisección por `git archive` ubica el origen en `014b422`: su padre `66951c5` pasa 80/80 y `014b422`, `659554e`, `431e44e` y `cdd3c84` fallan las mismas seis.

Causa con evidencia de referentes (`gc.get_referrers`, sólo observación, sin `collect`): el `Context` copiado por `asyncio.to_thread` conservaba el `NativeExecution`; el binding retiene la tarea terminada como testigo; la tarea guarda la excepción, cuyo traceback mantiene los frames con argumentos privados y el mismo `Context`. Es un ciclo inalcanzable que sólo libera el colector cíclico. Las seis fallas comparten esa cadena.

- `743efd3`: la ContextVar guarda un `_NativeSlot` compartido y revocable; el owner lo revoca junto al `_leave_native` existente, cuando el binding ya no valida. `_execute_native_step` limpia su `parent_sequence`, que hacía que la secuencia se retuviera a sí misma. Se rechazó antes un intento con referencias débiles a la tarea: rompía los testigos de finalización (43 FAIL) según dos revisiones independientes.
- `740ca42`: una sonda real sobre `EventRetention → OwnedBlockingWork → output_registration_v2` confirmó el mismo tipo de ciclo, también nuevo en `014b422`. Frames con `execution` (admisión, preflight, `_Operation`) quedaban en el traceback guardado. Con el worker `to_thread` por defecto, además, `_OPERATION` quedaba en el `Context` copiado. Se retenían metadatos de ruta/plan/grant y, en el caso `to_thread`, locals de filas de eventos: se registra como hallazgo de privacidad. `observe_native` limpia los locals de frames terminados de una llamada propia fallida (cadenas y miembros de grupos) después de cerrar la ventana de admisión. `_Operation.execution` pasa a referencia débil. Los testigos fuertes no cambian. Supone CPython ≥ 3.13.

Verificación local: los tres archivos originales 80/80. Las pruebas nuevas son RED en la revisión previa y GREEN tras la corrección, con GC automático suspendido y sin `collect` explícito. Los 66 archivos que tocan owned work dan 1302 PASS / 0 FAIL. Ruff 0.17.4 limpio en lo tocado, salvo `F401 MethodType`, preexistente desde `014b422` y no modificado. No se cambiaron aserciones, oráculos weakref/traceback, timeouts ni límites.

Separado y abierto: retención **preexistente** (también en `66951c5`) cuando el propio observador recibe la excepción nativa, vía el `Future` terminado. Los mapas de testigos (`_dispatches`, `_native_validations`) retienen tareas terminadas hasta la siguiente poda. Ambos quedan como trabajo propio.

Presupuesto de la full, investigado aparte: el orden no cambió y las seis fallas costaron +12 s. Las causas son +109 s de pruebas nuevas y +275 s de pruebas existentes, 85 % en cinco pruebas compiladas. El mismo código ya varió 202 s (`ef8bc6d`), y el margen a velocidad V1 sería de sólo ~85 s. Sin ampliar límites, la validación fresca round-02 separa las 7 pruebas compiladas en intentos nativos de 300 s cada uno, y la full corre el resto con `--deselect`. Un verificador prueba cobertura exacta y disjunta de IDs.

Parser JUnit, aparte: el elemento anónimo de la interrupción hacía fallar `validation_runner.py:84` (`KeyError`). Se perdían conteos/IDs aunque el estado fallido se conservó. El exportador contaba 5627−6=5621 elementos como PASS. Las versiones nuevas de round-02 cuentan sólo casos con nombre, separan los no identificados y contrastan los totales declarados. Los recibos de round-01 no se tocan y sus helpers de aceptación no se ejecutaron.

Validación fresca round-02 en `0fed9c4` (límites originales; sin otro pytest en paralelo). Colección exacta y partición disjunta de 9388 IDs: la full con 9381 más 7 nativas. Component dio 102 PASS. Las nativas 01–07 dieron 1 PASS cada una, entre 53 y 188 s de 300 s. La full dio 9380 PASS + la xfail regex conocida, 0 FAIL/error/no identificados, en 776,44 s de 1500 s. Las seis fallas originales pasan. Fuentes, protegidos, runner e inputs no cambiaron. Los nueve árboles crudos están exportados; la full incluye 15 FIFOs inertes. Una revisión independiente de la evidencia dio `accept_within_scope` en 7 comprobaciones. Dos firmas de warnings de terceros faltaban en round-01 sólo porque esa full se cortó al 59 %.

**Unidad aceptada dentro de su alcance**: `.../lifetime-cycle-validation-round-02/acceptance-proof.json`, SHA `49ccafb39f37648d06bc7de0d8c90b2a3b8c807b36c852cf5ebef42a9cc3e54b`. Evidencia de la reparación: `data/autonomous-roadmap-20261102/t024/f1/lifetime-cycle-repair/repair-evidence.json`. No acepta F1, T024, H6 ni el runtime, y no implica release. F1 y T024 siguen incompletos.

Limpieza de disco autorizada por el usuario: se borraron sólo 18 raíces `/private/tmp/kza-*` cuyos archivos coincidían por sha256 con su exportación. Las 13 parciales se conservan. El registro está en `lifetime-cycle-repair/tmp-root-cleanup.json`.

Siguiente: el usuario autorizó preparar, **sin instalar**, un paquete candidato de release. Incluye artefacto reproducible, manifiesto, riesgos, respaldos y rollback; la instalación queda a decisión del usuario y la ejecuta el operador. Después siguen cuentas/V3 conservando la prioridad T047.

Paquete preparado sobre `8b1575f`, reproducible desde un clon independiente. Veredicto **NO-GO**: el rollback a `63ea8d3` rompe tablas nuevas, y los recordatorios migran a `scope='private'` mientras el scheduler sólo dispara compartidos. Falta además una herramienta de publicación propia y un ensayo sobre una copia del estado de producción. Ver `data/release-candidates/8b1575f/risk-analysis.md`.

## Cuentas/V3 — publicador del catálogo de cuentas, aceptado

El registro de salida de producción escribe ahora la gramática V3. Contiene las familias V2 con sus comprobaciones V2 intactas, más el publicador `UserManager` de cuentas y un evento `composition`. El publicador se registra adjunto (`attach_accounts`) con el panel encendido, o como `disabled` sin panel. Los clientes V2 explícitos siguen escribiendo y leyendo V2. El decodificador V3 nunca reintenta un V3 mal formado como V2.

- `922b80c`: implementación.
  - `UserManager._save` admite la escritura antes de reservar el temporal. Transfiere el descriptor en dos etapas (`FileIO`, luego el wrapper de texto) para cerrarlo exactamente una vez, y borra sólo su temporal probado.
  - Las llamadas de administración se observan con su función y argumentos originales.
  - Los dos tests que esperaban V2 se migraron a V3 con autorización del usuario. Cambian los valores esperados: el decoder pasa a V3 y `schema_version` de 2 a 3. No se quitó ni se debilitó ninguna aserción.
- `5569b92`: corrige el pin de `composed(graph)`. `ApplicationGraph` no tiene `user_manager`; la comprobación pedía ese campo y rechazaba todo arranque real con panel. Ahora se fija a `graph.dashboard` (y a su `user_manager`), o a `None` sin panel. Lo detectaron los tests compilados, y un test de deriva protege el campo.
- `dae7c2d`: migración de los escenarios compilados.
  - **Compatibilidad:** prueba V1 (`ef8bc6d`) más V3 bajo un mismo controlador, con alta, cambio de permiso y baja reales por el panel dentro de `graph-v3`. El plan pedía también una rama V2 fijada, pero no es ejecutable. `activate` rechaza autoridad vieja (`privacy_authority_stale`, `stage_runtime.py:2299-2301`) y no existe delegación multi-salto. Además `b0866e8` no contiene ni `ef8bc6d` ni `c0e82df`, así que V2 nunca se publicó. El usuario eligió V1+V3, y el V1 del test es un fixture de compatibilidad, no el estado previo real del servidor.
  - **Límite:** el de fase de grafo de ese nodo pasó de 55 a 120 s, autorizado por el usuario. `graph-v3` con el paso 5 mide unos 77 s. Es un límite aflojado dentro de la suite validada. Los demás límites no cambian.
  - **Reconciliación de material:** pasa a esperar V3 en lugar de V2, `schema_version` 3 y el dueño `accounts` en cada conjunto, con igualdades exactas. `accounts` termina en `drained` y el resto de los dueños en `closed`. Su hijo cambia el permiso por `run_admin_work`, igual que la ruta real.

Hallazgos descartados con evidencia:
- **Cambio de permiso en el grafo compilado:** no se cuelga. Responde en 1,76 s y el proceso completo sale con exit 0.
- **Alta de voz por el panel con cuentas adjuntas:** no escribe nada. Responde 409, y `set_voice_embedding` devuelve sin cambios porque `voice_review_managed` es verdadero.
- **Doble cierre del descriptor:** el riesgo existía en el diseño de la primera ronda y se reprodujo en CPython 3.13.18. La transferencia en dos etapas lo evita.

Validación fresca round-03 en `dae7c2d`, sin ampliar límites de la ronda y arrancando cada suite con carga de 1 minuto menor que 10.
- **Partición:** 9459 IDs exactos y disjuntos: los 9388 de round-02 más 71 nuevos.
- **Resultados:**
  - Component: 173 PASS.
  - Nativas 01–07: 1 PASS cada una, entre 55 y 181 s de 300.
  - Full: 9451 PASS más la xfail regex conocida, 0 FAIL, en 823,5 s de 1500.
- **Revisión del coordinador:**
  - lo observado coincide con lo recolectado en los 9 modos;
  - 27 eventos de carga, sin solapamientos y sin arranques con carga ≥ 10;
  - árbol limpio y `HEAD == @{upstream}`.

**Unidad aceptada dentro de su alcance**: `.../accounts-v3-validation-round-03/acceptance-proof.json`, con revisión independiente `independent-review.json`: `accept_within_scope`, recalculada desde los archivos crudos. `36aa8af` registró la aceptación antes de que existiera esa revisión; el commit siguiente lo corrige. La nota de desvíos está en `data/autonomous-roadmap-20261102/t024/f1/accounts-v3/compiled/plan-deviation.json`. No acepta F1, T024, H6 ni el runtime, y no implica release.

Quedan abiertos:
- **Márgenes de tiempo:** `graph-v3` mide unos 77–99 s contra 120. La nativa de material mide 141–236 s contra 300.
- **Avisos de ruff:** los dos F401 preexistentes.
- **Release:** el paso real desde `b0866e8` sigue siendo un bloqueante del paquete candidato.

Siguiente: retomar el roadmap con la prioridad T047. T047 está `verified_local`; hay que precisar qué falta para aceptarla.

## Configuración/V4 — publicador de configuración en un journal V4 extensible, aceptado

`2f5e657` registra el escritor `ConfigurationChanges` de `state_root/config/settings.yaml` en el journal de salida. Pasa a la gramática V4, decidida por el usuario el 2026-11-09:
- **Tabla de gramáticas:** la cabecera declara `grammars: {productor: gramática}`, que se valida contra una tabla cerrada y sólo ampliable (`account_publisher/1`, `configuration_publisher/1`).
- **Gramática desconocida:** se lee como `unsupported_producer_grammar`, y el escritor no arranca sobre una corrida que la contenga.
- **Decodificadores anteriores:** V1–V3 no cambian.
- **Productores futuros:** cada uno debe estar ya en `base.PRODUCERS`.

Comportamiento del publicador:
- **Registro:** `attach_configuration` va después de `install_on`; `configuration_disabled` se usa sin panel o sin admisión.
- **Registro antes del archivo:** al cerrar, el registro se abre antes de la tarea escritora y de cualquier efecto sobre el archivo.
- **Si el registro falla (cierre ante el fallo, decisión del usuario):**
  - no se escribe `settings.yaml`;
  - se registra `registration_failure`;
  - no hay recibo de admisión, así que la admisión siguiente rechaza con `runtime_admission_not_closed`.
- **Registro terminal:** exactamente uno por cierre, también si la tarea escritora no arranca o no devuelve resultado.
- **`committed`:** exige un inodo nuevo. El hash de `settings.yaml` se comprueba al adjuntar y al leer.
- **Qué escribe:** el escritor sólo pone `reasoner.cloud.consent = false`. Si el registro terminal falla después de `committed`, el archivo ya queda sin consentimiento.
- **Orden de cierre:** se dedujo del orden de registro en `src/main.py`, no se midió con un test dedicado. `install_on` se apila (línea 408) antes que `http_reasoners`/`hermes_reasoners` (486/488), así que los reasoners cierran antes. En ejecución, `_require_graph_closed` exige además que privacidad haya cerrado del todo.

Tests:
- Los tests V3 se migraron a V4 y `graph-v3` se renombró `graph-current`. Esto cambia valores esperados y etiquetas internas, no node IDs.
- Se quitaron 13 aserciones, cada una con su equivalente V4, y se agregaron 5 nuevas. Ninguna se debilitó.
- Los 29 mutantes de la revisión del diff (R1–R9) mueren.

Limitaciones conocidas:
- **Sin caller de retiro:** hoy ningún código de producción llama a `queue_consent_withdrawal`, así que no hay retiro en vivo.
- **`EgressPolicy` inmutable:** es `frozen`, así que el gateway sólo se apagaría desde el siguiente arranque. El usuario eligió para OQ-8 un marcador durable registrado en el journal más una compuerta en ejecución sobre `EgressPolicy.require_allowed`, que es el único punto de intercepción. Es la tarea siguiente.
- **R8:** un `settings.yaml` con hardlink o editado entre admisión y adjunto bloquea el arranque. Un rollback a un release que no conozca una gramática nueva también lo bloquea.
- **R9:** la nativa 06 ya no lee un historial V3 genuino. Lo cubren 12 tests no compilados.

Validación fresca round-04 en `2f5e657`, con los mismos límites y una compuerta de carga de 1 minuto menor que 10 y sin otro pytest.
- **Partición:** 9551 IDs exactos y disjuntos: los 9459 de round-03 más 92 nuevos (53 de configuración y 39 de V4), y 0 eliminados.
- **Resultados:**
  - Component: 275 PASS.
  - Nativas 01–07: 1 PASS cada una, entre 46,6 y 171,1 s de 300. La 06 mide 99,3 s, frente a 98,2 s en round-03.
  - Full: 9543 PASS más la xfail regex conocida (el mismo ID de round-03), 0 FAIL, en 801,3 s de 1500.
- **Revisión independiente:** `independent-review.json` (`accept_within_scope`), escrita **antes** de la prueba de aceptación.

Hallazgos no bloqueantes de la revisión:
- los scripts de `root-procedure/` sólo se pueden rastrear hasta una copia en `/private/tmp`; hay que fijarlos antes de la próxima ronda;
- el pico de carga de 242 después de native-01 no tiene evidencia de procesos guardada;
- las colecciones no figuran en `load-observations.jsonl`;
- la carga sólo se registró al inicio (3,22) y al final (4,99) de la full, sin registro intermedio, así que sus 801 s pueden estar afectados por carga externa.

**Unidad aceptada dentro de su alcance**: `.../configuration-v4-validation-round-04/acceptance-proof.json`. No acepta F1, T024, H6 ni el runtime, y no implica release.

Siguiente: OQ-8, con el marcador durable en el journal y la compuerta de egress en ejecución, y después la prioridad T047.

## OQ-8 — retiro del consentimiento cloud: marcador durable y compuerta en ejecución, aceptado con salvedades

`0c8f5be` implementa OQ-8 y `0e9cc8a` corrige una migración de test. Decisiones del usuario (2026-11-09 y 2026-11-10):
- **Alcance:** marcador más compuerta. El marcador vive en el journal, no en un archivo de config.
- **Endpoint:** se agregó ya, sólo con `admin:manage`, sin acceso por voz ni persona.
- **Reversión:** se mitiga por runbook.
- **Journal ilegible al arrancar:** cuenta como retirado.
- **Pedidos en curso:** se cortan.
- **Router:** cae al 7B local.
- **Arranque tras un cierre no limpio:** si el journal se lee y no tiene marcador, manda `settings.yaml`.

Implementación:
- **Compuerta:** `CloudConsentGate` es compartida y se inyecta por constructor. Se comprueba:
  - en `EgressPolicy.require_allowed`;
  - en un hook por pedido HTTP, con `max_retries=0`;
  - antes de cada fragmento de streaming.
- **Clientes cloud:** uno con consentimiento no se construye sin compuerta (`consent_gate_required`).
- **Retiro:** `POST /api/admin/cloud-consent/withdraw` corre como una única tarea retenida, en este orden:
  1. autoridad;
  2. cierre de la compuerta;
  3. marcador durable `configuration_publisher/2`;
  4. escritura de cierre encolada.

  Cualquier fallo después de cerrar la compuerta envenena la época: no hay recibo limpio. El cierre espera un retiro en curso antes de leer el pendiente (N1).
- **Arranque:** lee la decisión del run legible más nuevo, que va en `intent`. Un run más nuevo ilegible se registra como `unverifiable` y se arrastra hasta un arranque que encuentre `consent: false`.
- **Hermes:** se mata el grupo de procesos. Un proceso lanzado durante el retiro también muere (F2/N3).
- **`/1`:** queda idéntico. Lo verifiqué con las 29 definiciones de primer nivel y con 150 tests propios.

Validación:
- **Round-05 (`0c8f5be`), fallida:** `test_stage_runtime` construía un reasoner con consentimiento y sin compuerta. Se corrigió en `0e9cc8a`.
- **Round-06 (`0e9cc8a`), fallida:** la full se colgó unos 24 min en `test_capture_native_clients.py::test_diagnostics_has_its_own_capability_and_three_native_ack_barrier`. La causa sigue desconocida y no se reproduce sola, con su módulo ni con el mismo prefijo. Investigación: `data/.../oq8/round06-hang-investigation.json`.
- **Round-07 (`0e9cc8a`), verde:**
  - partición de 9756 IDs (9551 más 205, 0 eliminados);
  - component con 275 tests previos más los nuevos, 493 en total;
  - nativas con 1 PASS cada una;
  - full con 9748 PASS más la xfail conocida, en 863 s de 1500.
- **Cambios de procedimiento:**
  - el runner captura stacks (`faulthandler_timeout=120` y `sample` al vencer el plazo);
  - la compuerta de carga se quitó por decisión del usuario desde native-03;
  - los límites, aserciones y GC no cambiaron.
- **Revisión independiente:** `accept_with_caveats`.

**Aceptado dentro de su alcance, con salvedades**: `.../oq8-validation-round-07/acceptance-proof.json`. No acepta F1, T024, H6 ni el runtime, y no implica release.

Salvedades y pendientes:
- **Defecto abierto:** el cuelgue intermitente de round-06. No se encontró un camino causal desde OQ-8, pero tampoco está probado que sea ajeno. Una sola corrida verde no acota su frecuencia.
- **K5:** code-index y `sync_ha_to_chroma.py` crean su propia compuerta en otros procesos, así que un retiro no los alcanza.
- **Cierre a mitad de un retiro:** el marcador queda durable pero `settings.yaml` no se reescribe. La nube sigue apagada hasta que un operador fije `consent: false`.
- **Reversión:**
  - `b0866e8` ignora el marcador. Antes de revertir hay que verificarlo y fijar `consent: false`.
  - `2f5e657` no arranca si existen runs `/2`.
- **Antes de H6:** falta el texto de runbook en `docs/architecture/DEPLOYMENT.md` (F11/F15/F16 y la precondición de reversión D7).

Siguiente: la prioridad T047.

**Actualización 2026-11-10 — el cuelgue de captura quedó resuelto** (`7b24460`). `CaptureRecorders.drain` y `aclose` repetían `while self._operations: await gather(...)`. Pero `gather` sobre tareas ya terminadas resuelve sin ceder el event loop, y el callback que las saca del conjunto queda encolado con `call_soon`. Si una operación terminaba justo antes de `drain`, el bucle giraba sin fin. El stack de faulthandler de la ronda 01 de T047 lo ubicó en `capture_recorders.py:291`. El arreglo espera una foto del conjunto y descarta esas tareas.
- **Pruebas:** una prueba determinista roja→verde (`tests/unit/memory/test_capture_recorders_drain.py`) y el módulo nativo 20 de 20 seguidas.
- **Revisión independiente:** `accept`. Evidencia en `data/autonomous-roadmap-20261102/t024/capture-drain-hang/`.
- **Pendiente:** hubo una sola full verde desde el arreglo; conviene confirmarlo en más corridas.

## T047 — NEXO en el frontend real, aceptado localmente dentro de su alcance, con salvedades

Decisiones del usuario (2026-11-10, `data/.../t047/user-decisions-20261110.json`):
- **Aceptación:** acotada a su alcance.
- **Vistas heredadas:** se retiran `/home/legacy.html` y el enlace al panel técnico, en T040/T041.
- **Casa 3D:** la casa 3D ilustrativa alcanza.

Ronda 02 sobre `7b24460` (`data/.../t047/validation-round-02/`), sin reintentos:
- node 178/178;
- navegador NEXO (home, voces, accesos) con 0 hallazgos de accesibilidad en 320–1440 px;
- pytest dirigido: 850 passed;
- full: 9750 passed más la xfail conocida, 0 fallos, en 941 s de 1500;
- referencia de diseño: 22/22.

Revisión independiente: `accept_scoped_with_caveats`. Prueba en `data/.../t047/acceptance-proof-20261110.json`. El estado de la tarea en el roadmap sigue `verified_local` y tiene un nuevo campo `scoped_acceptance`. **No acepta** C02/C04/C06/C07/C08/C18 ni P4.

Pendiente explícito:
- **T040/T041:** WEB-01/02/05 y ACCESS-01; retirar las vistas heredadas y el enlace técnico.
- **Campo:** iPhone físico; Chrome en Terra; usabilidad 3×10; lector de pantalla; voces reales; privacidad en campo; matriz negativa C02; catálogo real de Home Assistant y efectos observados.
- **Accesibilidad de diálogos:** botones de 40 px en teléfono en el diálogo de revisión de voces. La auditoría no abre diálogos.
- **Otras auditorías:** WCAG estándar (T042) y matriz de errores C06 (T045).
- **Fidelidad visual:** comparación contra las previews y visto bueno del usuario.
- **Suites compiladas y release (H6).**
- **Separación voz/persona respecto de permisos:** observarla en ejecución con cuentas reales. Hoy está verificada por traza de código y pruebas sintéticas.

La referencia de diseño en el Mac está en `/Users/gabriel/Documents/Proyectos/kza-design-reference/nexo-2026-11-02-d686fcd`.

Siguiente: T040/T041 (retiro de vistas heredadas), luego los productores F1 restantes y el runbook OQ-8 en DEPLOYMENT.md antes de H6.

## T040/T041 — retiro de vistas heredadas (2026-11-10)

Secuencia de `data/autonomous-roadmap-20261102/t040-t041/user-decisions-20261110.json`:
U1 (`2873fbc`) quitó el enlace a `/obs/` y arregló sus controles; U2 (`cfa985a`) y
U3 (`5af9b5c`) agregaron Rutinas y Memoria a NEXO. U4 retira en desarrollo
`/home/legacy.html`, `home.css`, los `routines.js`/`memory.js` heredados, el
dashboard raíz (`/index.html`, `dashboard.js`, `style.css`), el enlace
"Micrófonos y diagnóstico" y `#additional-tools`. Con `dashboard.home.enabled`
falso, `/` responde 404 con una página que nombra esa opción. Evidencia en
`data/autonomous-roadmap-20261102/t040-t041/u4-evidence.json`. La release
instalada (slot activo 63ea8d3) no tiene `legacy.html`: sus vistas heredadas son
`/home/` y la grilla raíz `/index.html`, y las conserva hasta la próxima release.
Revisión independiente: `u4-independent-review.json` (accept_with_changes). Abierto:
F3, los controles de una entidad se deshabilitan un instante porque la observación
vence a los 6 s y el sondeo del catálogo tarda más de eso bajo carga.

## T040/T041 — aceptadas localmente con salvedades (2026-11-10)

Unidades:
- **U1** (`2873fbc`): `/obs/` queda accesible solo por URL directa, con controles reales o deshabilitados honestamente.
- **U2** (`cfa985a`): Rutinas en NEXO.
- **U3** (`5af9b5c`): Memoria en NEXO, solo administración.
- **U4** (`926fc0a`): retiro de las vistas heredadas.
- **Arreglo de frescura** (`a69039a`): el catálogo se consulta contra el vencimiento de las observaciones.
- **U3b-0a** (`69a79e6`): `activate` sondea el slot candidato con su propio controlador y agrega el comando `delegate` de un salto.

Ronda 01 sobre `69a79e6` (`data/.../t040-t041/validation-round-01/`):

| Prueba | Resultado |
|---|---|
| node | 275/275 |
| Navegador | 10/10, 0 hallazgos de accesibilidad |
| Dirigido | 1240 |
| Full | 9822 + xfail conocida, 0 fallos, 886 s |

Es la segunda full verde tras el arreglo del cuelgue de captura (`7b24460`). Revisión independiente: `accept_scoped_with_caveats`. Prueba: `data/.../t040-t041/acceptance-proof-20261110.json`. El estado sigue en `verified_local`; no acepta C04/C06/C07/C08 ni P4.

Estado de los hallazgos:

| Hallazgo | Estado |
|---|---|
| WEB-01 | Cerrado localmente con salvedades |
| WEB-02 | Cerrado localmente con salvedades |
| WEB-05 | Parcial: la vista de `/obs/` no queda en la URL y `/obs/` a 320/390 px no se auditó |
| ACCESS-01 | Abierto (T013); el login de `/obs/` menciona el token |

Otros pendientes:
- Memoria: paginación (límite de 100) y mapeo 404/409 posterior al commit (F3).
- Alcance del borrado C04.
- Q19, a cargo de T033.
- Permiso `routines:execute` de hogar, que no habilita nada.
- Falla de `smoke_dashboard` en `/ws/live`.
- Visto bueno visual del usuario.
- Campo P7.

Memoria por persona (U3b): las decisiones del usuario están en `user-decisions-20261110.json` (`u3b_*`):
- asignación por administración más voz reconocida con alta propia;
- administración ve todo;
- asignar confirma el recuerdo;
- no volver a `b0866e8` después de U3b;
- «No es mío» lo devuelve a una cola de administración;
- la voz se compara contra todos los perfiles solo para descartar;
- el esquema sale en una release anterior a la función.

Diseño v5 más enmiendas de 0a en `data/.../t040-t041/`. Siguiente: U3b-1a (núcleo del ledger), aplicando los cambios que la revisión del v5 exige para 1a.

## U3b — memoria por persona: núcleo, baja y API (2026-11-10)

Commits:

| Commit | Contenido |
|---|---|
| `69a79e6` (U3b-0a) | `activate` sondea el slot candidato. |
| `58d7b5c` (U3b-1a) | Tablas de dueño. Asignar confirma; regla de derivados; Devueltos; ámbitos; protección en restauraciones; registro en admisión. |
| `8f81711` (U3b-1b) | Dar de baja una cuenta borra sus recuerdos asignados; purga al arrancar. |
| `5f287eb` y `4074a3e` | La derivación desde un estado pre-U3b acepta el esquema de dueño vacío. Los ensayos de slots parten de una versión con el mismo registro de esquema. Hay un ensayo dedicado para la cobertura pre-014b422/pre-ef8bc6d. |
| `da60173` (U3b-2) | Permisos `memory:own:*` y `memory:assign`, rutas `/api/memory/mine`, asignación y Devueltos, paginación keyset, matriz de 312 denegaciones, purga con autoridad desconocida = pendiente. |

Validación: `data/.../u3b/validation-round-01/`, verde sobre `da60173`.

| Prueba | Resultado |
|---|---|
| node | 275 |
| navegador | 10/10 |
| dirigido | 2241 |
| full | 10210 + xfail |

Es la tercera full verde tras el arreglo del cuelgue de captura.

U3b-3 (`238b463`): pantallas NEXO (Mis recuerdos, asignación, Devueltos), smoke compilado 13/13.

U3b-0b: delegación de varios saltos en `scripts/stage_runtime.py` (cadenas en `admission/delegation-chains/`, tope 8 dueños: el 9.º se rechaza). Revisión independiente `accept_with_changes`; F1–F7 corregidos. Abierto:
- F8: un borrado de una fila protegida en un estado dueño deja el head `unknown`. Ya existía con un salto; queda como xfail estricto. Requiere diseño de re-vinculación.
- Binding de más de 16 MB (unas 250k filas protegidas) rechazado como `unsafe_record`. Viene de 0a.
- Tope de dueños: el usuario eligió **compactar** (2026-11-11). Unidad nueva antes de la primera release gestionada en producción.
- Falta medir con un estado del tamaño de producción.

U3b-4 (voz, sintético y local): diseño v4 en `t040-t041/u3b-4-design-v4.json`, tres revisiones independientes. Decisiones del usuario Q1–Q7 en `u3b_4_voice`:
- R1 sale con todo U3b y la voz apagada.
- Antes de que la administración confirme, la persona ve un aviso sin texto, con «Descartar» y «No es mío».
- Con una sola voz registrada no se atribuye nada hasta P7.

`389a18c` (U3b-4a-0): tablas de voz protegidas e interruptor de administración (G1). La fila faltante significa apagado y un cambio del interruptor cambia el digest de autoridad. Revisión `accept_with_changes`.

P0 confirmó que editar `settings.yaml` del estado se rechaza con `privacy_authority_stale`. Prender la voz requiere un procedimiento nuevo de cambio de configuración; apagarla usa G1.

`293169d`: la búsqueda en memoria filtra solo por los IDs que el índice tiene. Arregla el fallo intermitente `memory_prompt_unavailable` (índice atrasado respecto del registro). La espera es acotada a 5 s y después da `memory_search_busy`.

`d513dc2` (U3b-4a-1): consentimientos de voz.
- La solicitud exige exactamente un perfil vinculado.
- El bloqueo sobrevive a la revocación y solo lo reabre un operador.
- La verificación tiene un límite por día contado sobre filas guardadas.
- La comparación solo rechaza.
- La atribución tiene dos pasos y relee todas las revisiones.
- Las propuestas pendientes no muestran texto; incluyen «Descartar» y «No es mío».
- Por ahora los puntajes son inyectados; el puntaje real llega en 4b.
- Revisión `accept_with_changes`; A–I corregidos y 66/66 mutantes detectados.

U3b-4a-2: permiso `memory:own:voice`, rutas de la persona y de administración, y matriz de 211 filas. La verificación no permite sondear el estado del micrófono. Revisión `accept_with_changes` (R, W, T, L, C, P), todo corregido; 74/74 mutantes muertos.

La suite completa ya no entra en una corrida de 1500 s, así que se corre en dos tandas con el mismo límite (ver `u3b/full-4a1-r3/RESULT.md`).

U3b-4a-3: pantallas NEXO de voz.
- En «Mis recuerdos»: inscripción, verificación y una tarjeta sin contenido con «Descartar» / «No es mío».
- En Voces: interruptor, consentimientos, reabrir y reanudar.
- La revisión fue `accept_with_changes`; S (marcas borradas en pagehide), T, K y Z ya están corregidos.
- R2 queda con aviso: el cliente no puede saber qué habitaciones tienen micrófono sin un campo nuevo del servidor.
- Pendiente con el usuario: el visto bueno visual y dos textos que difieren del diseño v4 («Todavía no se puede verificar tu voz en este hogar…» y la línea con la voz apagada). Capturas en `t040-t041/u3b-4a-3-dev/browser-runs/`.

`cd8bc47` (U3b-4b): puntaje sobre embeddings guardados, atribución tras la respuesta fija y frases de verificación.
- Las frases se filtran contra confusiones de ASR y contra los nombres de Home Assistant.
- Los casi-aciertos se absorben mientras haya una ventana abierta.
- La activación textual ya no loguea texto.
- La revisión fue `accept_with_changes` (L1, L2, W, E, T); todo quedó corregido.
- 4b-3 se saltó porque requeriría speechbrain y descargas.

Antes de prender la voz (4c):
- calibrar los umbrales con voces reales después de P7, contra la meta de falsos aciertos acordada con el usuario;
- verificar el codificador real en el servidor;
- contar con el procedimiento de cambio de configuración, porque P0 mostró que editar `settings.yaml` se rechaza.

Huecos de backend (`c343b94`):
- El recibo de asignación incluye `deleted_dependent_ids`.
- `whoami` devuelve `account_binding`, una marca opaca que no otorga acceso.

Privacidad: hay copias que el borrado no alcanza. Está documentado en DEPLOYMENT (`04db45f`).

Lanzador y compactación:
- Diseños revisados: `managed-launcher-design-v3.json` y `owner-compaction-design-v5.json`.
- Decisiones del usuario: `u3b_0c_*` y `u3b_0d_launcher`.
- Orden previo a R1: 0d1 → 0d2 (bucle sin bloqueos) → 0d3 (registro de borrados) → 0d (lanzador) → 0e (adopción) → 0c (compactación). F8 va antes de la segunda release gestionada.

`U3b-0d1`: contrato de parada y readiness. Antes de instalar 0d falta:
- correrlo en Linux;
- medir en el server el presupuesto de parada.

La excepción sudo `zz-kza-stop-timeout.conf` la aplica el usuario. La causa de los 5 s es el `timeout.conf` del paquete systemd de Ubuntu.

`cc73f39` (U3b-0d2): el bucle principal queda sin bloqueos y el índice de comandos se reconstruye de forma segura (staging + puntero). El rollback conserva la colección legacy.

U3b-0d3: registro de borrados sin contenido (7 tipos) y resúmenes del estado en `admission/receipt-states/`. Prerrequisitos para 0d:
- K2: release.json declara que el sidecar es obligatorio.
- K7: el presupuesto de parada cuenta 3 snapshots.

La suite completa corre en 3 tandas: unit, integration y unit/tools. `40aa05b` quedó verde; los 5 ensayos compilados con presupuesto de tiempo pasaron solos (`u3b/full-0d3-r2/RESULT.md`). Esos ensayos corren sin otro pytest en paralelo.

U3b-0d: lanzador gestionado (`6288e15`, `55cdc42`, `a96780d`, `9cf0b1b`).
- systemd arranca kza-voice a través del controlador; SIGTERM cierra con recibo; recuperación desde UNKNOWN (una por 24 h en crash, una por arranque tras reinicio).
- Decisiones nuevas: L7 (componentes manuales distintos ⇒ siempre rechazar, aun con operador) y L8 (la recuperación por reinicio no consume el cupo de 24 h).
- Revisión independiente: P1–P11 y P14 corregidos (`t040-t041/u3b-0d-fix-evidence.json`); P12/P13 documentados en DEPLOYMENT. 28/28 mutantes muertos.
- Validación: `tests/unit/tools` 651 passed + 1 xfail; ensayos compilados del lanzador 34 passed, 1 skip opcional (`u3b/u3b-0d-import/`).
- Antes de instalar: T-C8 y T-C10 en Linux, medir R-b en el server, aplicar la excepción sudo en la misma ventana que R1, y U3b-0e.

U3b-0e: adopción del estado de producción como primer estado gestionado.
- Diseño `t040-t041/u3b-0e-adoption-design-v3.json`, tras dos revisiones independientes (v1: F1–F19, v2: N1–N12).
- Decisiones del usuario P1–P8 en `u3b_0e_adoption`:
  - los LoRA se retiran en la ventana;
  - `audit.db` y la tabla `events` vieja no se llevan, y la auditoría queda apagada en R1;
  - los borrados a medias quedan anotados como pendientes;
  - la vuelta atrás es a 63ea8d3; sin frase solo antes de la primera corrida gestionada;
  - la copia vieja se retira 14 días después del primer cierre limpio;
  - los limpiadores faltantes van después de R1 (0d4b).
- Orden: 0d4a → 0e1 → 0e2 → 0e3 → 0e4 (Linux) → 0e5 (Q5 + chequeo del destino de vuelta, antes de agendar R1).

`U3b-0d4a` (`1df524b`, `196f5cc`, `c62e38b`): detección de conversaciones borradas que reaparecen, por almacén. Antes, todo almacén restaurado contaba 0.
- Revisión independiente: F1–F3 y F5–F7 corregidos; F4 y F8 pasan a 0e2/0e3. 42/42 mutantes muertos.
- Un archivo de colector restaurado sin su fila deja el paso pendiente hasta 0d4b; se ve en `conversation_exempt`.

`U3b-0e1` (`836d5a8`..`ab7ad8c`): código de R1 para la adopción.
- `paths.models_root` y su lint.
- `hooks.audit_sqlite.enabled`, apagado por defecto y rechazado bajo ticket.
- G1: no se arranca sin gestionar dentro de una raíz gestionada.
- Runtime de solo lectura.
- Hijo de solo esquema limitado a copias de adopción.
- Revisión independiente: F1–F10 corregidos. 75 mutantes muertos.

Validación por tanda 0d4a + 0e1: 11223 passed, 2 xfail y 1 skip opcional; los ensayos compilados de presupuesto corrieron solos (`u3b/full-0e1/RESULT.md`).

Integrados y publicados hasta `2ad44c0` (2026-11-13):
- `U3b-0e2` (`5dfdc77`, `622ec38`, `2c1ad06`): biblioteca de adopción. Copia por bytes sin abrir el origen, migración de solo esquema con el código de R1, baja de la tabla `events` vieja con VACUUM, reubicador y sembrado del registro de borrados. Los restos se reabren en el paso que los gobierna. Revisión independiente: aceptada con hallazgos; quedan abiertos F2 (destinos con rutas anteriores al 28-10) y F4 (pasó a 0e3).
- `U3b-0e3`: comandos de operador. `retire-legacy-training`, `adoption-dry-run`, `adoption-prepare` y `adoption-commit` (este con intento de commit recuperable en cada punto de corte), `adoption-abort` y runbook `docs/runbooks/2026-11-13-r1-adoption-window.md`. Tres rondas de revisión independiente; R2-1..R2-4 corregidos.
- `U3b-0e5`: `retire-legacy-state` (Q5) y chequeo del destino de vuelta. Dos rondas de revisión.
- Arreglos de Linux, encontrados en la primera suite completa en el server como `yo`:
  - Defecto real de producción: los publicadores de configuración y de cuentas borraban un temporal sustituido cuando ext4 reutilizaba el número de inodo. Ahora mantienen un descriptor abierto hasta limpiar.
  - Destinos fijados en `output_registration_v2`.
  - Modos explícitos 0700/0600 en la raíz gestionada.
  - Fixtures portables con umask 0002.
  - Revisión independiente: aprobada.
  - Pendientes: F4 (SQLite comparado por inodo, con `O_PATH` en Linux); F5 (el servicio corre con `UMask=0002` y `audio_health.json` lo lee otra cuenta); F10 (limpiezas por ruta). También, un cierre de borrado de 1 s que vence con carga. (Resueltos después: F4/F10 en `f4-f10-identity`; F5 y el cierre, en la entrada F5 más abajo. Ninguna otra cuenta lee `audio_health.json`.)
- Suite por tanda en `2ad44c0`, en la Mac y en el server: verde salvo una prueba sensible a carga en la Mac (`u3b/full-linux-fixes/RESULT.md`). En el server, a2b tardó 1399 s; partirla antes de la próxima tanda allí.

Pruebas en el server: el usuario autorizó correrlas allí como `yo` y detener servicios si hace falta (`server_test_runs`). El checkout anterior de `yo` lo había borrado la limpieza del 2026-11-09; el nuevo está en `/home/yo/projects/kza`, con venv Linux y runner `/home/yo/projects/runs/run-server.sh`.

`U3b-0e4` (`8d89cf9`..`440617e`): ensayos compilados en Linux real, en el server como `yo`, con el systemd de usuario real. Las unidades son de prueba (`kza-rehearsal-*`) y se barren al terminar. Están en `tests/integration/test_linux_managed_launcher.py` y `test_linux_adoption_window.py`; corren sólo en Linux con systemd de usuario, de a un módulo por vez.
- Defectos de producto encontrados y corregidos:
  - `$VAR` dentro de un argumento de unidades estándar de Ubuntu (como IBus) trababa W1b. Con la corrección de la revisión, sólo se tolera en unidades que no se nombran y no tienen relación con kza.
  - El orden de los timers.
  - Un borrado pendiente del índice exceptuaba también la fila del hecho.
- `TimeoutStartSec` por defecto: 240 s, validado (30–3600) y leído de vuelta de systemd (decisión `u3b_0e4_start_timeout`: valor amplio + regla de no hacer `daemon-reload` mientras kza-voice arranca).
- Nuevo comando de solo lectura `completion-status` para después de W9.
- Medido sobre un estado de tamaño de producción (161 MB): ensayo 22 s, preparación 24 s, confirmación 6 s, primer arranque 12 s, arranque con R-b 20 s, parada 8,5 s.
- El swap del server está lleno (8 GB): el aviso de exposición a swap del ensayo se va a disparar.
- Huecos: T-A12/14/18 (que la primera corrida real complete los pasos) necesitan la aplicación real; W7 (sudo) no se aplicó.
- Revisión independiente: aceptada con arreglos (F1–F7, F11 corregidos).
- Suite completa en `440617e`, en la Mac y en el server: verde sin fallas (`u3b/full-0e4/RESULT.md`). En el server, la parte de integración ahora va en tres tandas.

`U3b-0c` (integrada en `a0209f2`, sobre `b3cbcd0` + `a526538`): compactación de estados dueños, la última unidad antes de R1. Runbook: [2026-11-14-owner-compaction](2026-11-14-owner-compaction.md); sección nueva en DEPLOYMENT.
- `stage_runtime compact plan|start|verify|retire|resume|abort|allow-run|status`. El estado nuevo K es una copia idéntica del head con las rutas reubicadas; tras una corrida gestionada limpia de K se destruyen con `unlink` sólo las copias con prueba positiva. Lo demás queda intacto y se reconoce por digest.
- Época del controlador 3; compuerta en la entrada del controlador y en `prepare`; serve deja pasar sólo la corrida armada o un `allow-run`; R-b automático apagado mientras haya compactación pendiente; head manual nuevo `compaction_return`.
- La fase de datos corre en un proceso aparte con su capacidad G1 (`KZA_COMPACTION_WORKER=1`).
- P14 en el server con el systemd de usuario real de `yo` encontró un hueco: una unidad deshabilitada e inactiva no aparecía en el listado del diseño (E2). Ahora se listan todos los estados.
- Mutantes: los 31 del diseño (v2, v4, v5), todos muertos, cada uno en su copia.
- Huecos registrados en `t040-t041/u3b-0c-evidence.json`: sin corpus de reloj envejecido completo; canarios no en cada familia de datos; G2 medido con 20 000 filas, no 1 000 000; `compact verify --wait` no implementado.
- Revisión independiente (`t040-t041/u3b-0c-independent-review.json`): «arreglar antes de producción». Decisiones `u3b_0c_review` D1–D4 = A. Corregido en la misma rama:
  - F1/F2: un corte entre el CAS de retorno y `returned`, o entre el `rename` de K y su registro de `prepare`, ya no traba la raíz (registro de retorno y registro de staging como prueba; `current` se vuelve a publicar hacia H). Inyección de cortes en cada borde de `abort` y de C2a.
  - F3/D4: `install-controller` (época ≥ 3) entra con una compactación pendiente o un log inválido; `compact status` informa un log ilegible con su código; sección «Compactación trabada» en el runbook.
  - F4: una corrida armada que falló antes del CAS de `begin_run` se puede abortar; `allow-run` rechaza `not_verified`.
  - F5/D2: el retiro mide la cobertura contra H; lo que borró la corrida de K cuenta como `removed_by_k_run` y no bloquea; datos nuevos exigen `retire --accept-owner-only-loss <coverage_sha256>`.
  - F6/F7: un slot es miembro sólo con exactamente `runtime/` y `state/`, y con el `state/` del plan; el retiro recalcula el plan y rechaza `plan_changed`.
  - F8/D1: comprobación de época hacia adelante en la entrada del controlador y en `prepare`; R1 se construye con U3b-0c (época 3).
  - F9: P14 resuelve `$VAR` con `show-environment`. Una referencia sin resolver sigue la regla de U3b-0e4 (`mentions_markers`, compartida): se tolera sólo en unidades no nombradas que no mencionan kza, un miembro ni la raíz (pasan IBus, podman y unity de la cuenta `kza`); `compact plan/start --unit` nombra más unidades. F10/D3 y F11 documentados. F12: rechazos con código en vez de tracebacks.
  - Huecos de la revisión cerrados: T-U10 (un borrado pendiente en H sobre un target heredado lo ejecuta la corrida de K sobre la copia reubicada) y mantenimiento K' vs H (el barrido TTL encuentra el clip reubicado).
- Segunda revisión independiente (`t040-t041/u3b-0c-independent-review-r2.json`, sobre `467cf02`): «arreglar antes de producción», N1–N10. Corregido en la misma rama, cada uno con prueba y mutante (sus sondas son ahora `test_stage_runtime_compaction_review_r2.py` y pruebas de P14):
  - N1: serve rechaza desde `abort_started` (`abort_pending`) y `retire_started` (`retire_in_progress`); si el head cambió después de `abort_started`, `abort --to-h` vuelve a aplicar Q6 y el descarte (`abort_rechecked`), y `resume` rechaza `abort_recheck_required`.
  - N2/N9: números estables por evento (K con número fijo) y placeholder reservado antes del `mkdir`; un abort reanudado termina lo que quedó en la cuarentena y no registra `aborted`/`complete` con residuos.
  - N3: los enlaces de restos `.activate-*`/`.prepare-*` se borran como enlaces. N8: un miembro movido tras el plan rechaza `member_missing` (`--accept-missing`: sin lápida). N7: contenido probado de copias del catálogo y de `runtime/`; slot con sólo prueba de staging requiere `--retire-prepare-leftovers`. N10: copia en preparación con dueño vivo rechaza; abandonada se lista.
  - N4/N5/N6 (P14): unidades nombradas siempre, enlazadas y plantillas de los directorios del usuario; valores de `show-environment`; marcas de home con prefijo (también en 0e4). systemd 255 real en el server: una unidad enlazada informa el symlink de `~/.config/systemd/user` como FragmentPath; `show` de una plantilla falla, y una instancia sin cargar no aparece en `list-units`.
- Tercera revisión (`t040-t041/u3b-0c-independent-review-r3.json`, sobre `f9f101f`): «arreglar primero», N1–N10 confirmados cerrados. Corregido, con prueba y mutante por hallazgo (sondas en `test_stage_runtime_compaction_review_r3.py`, `test_compaction_units.py`, `test_adoption_commands.py`):
  - R3-1: `compact abort <id> --member n --accept-missing` (sin lápida). R3-4: evento `renamed` después del `rename`; `retiring` solo no prueba un borrado.
  - R3-2: plantillas con continuaciones, `$HOME`/`${HOME}`, `Environment=`/`EnvironmentFile=` y palabras no juzgables (rechazo salvo tolerancia 0e4). R3-6: valores `$'...'` decodificados. R3-5: `kza*`, `~kza/`, `"$HOME"/`, relativo desde el home (también 0e4). R3-3: dueño de copia por pid y hora de arranque. R3-7: runbook, salida de `abort_revives`.
- Cuarta revisión (`t040-t041/u3b-0c-independent-review-r4.json`, sobre `aed3b6b`): «arreglar primero» (R4-1 y R4-2 requeridos; con ellos, lista para R1). Corregido, con prueba y mutante por hallazgo (sondas en `test_stage_runtime_compaction_review_r4.py`, `test_compaction_units.py`, `test_adoption_commands.py`, caso (i) de `test_linux_compaction_units.py`):
  - R4-1/R4-3/R4-4: regla de subcadena en plantillas y entorno del gestor (raíz o miembro en cualquier lugar; trozos cortados en caracteres del shell, `=`, `,`, `:` y juzgados por su parte fija); `EnvironmentFile=` gana sobre `Environment=` y sus valores se comprueban; variable en un `sh -c` de una plantilla de kza rechaza; `check_value` corta en `,` y en cada `=`.
  - R4-2: K movido a mano tras construirlo rechaza `member_missing` y se acepta con `--member n --accept-missing`. R4-6: evento `walked` (con la cuarentena vacía) es la única prueba de borrado; un aceptado que vuelve rechaza `accepted_member_back`. R4-5: enlace del home (`/home/kza/app`), `//`, `/./`, `${HOME:-…}`, `WorkingDirectory="%h"`. R4-7: `ps` en UTC en el Mac. Runbook: contradicción sobre `$'...'` corregida.

- Comprobación de cierre (`t040-t041/u3b-0c-independent-review-r5.json`, sobre `73e20f3`): **lista para R1**. R4-1 y R4-2 cerrados; con el entorno exacto del gestor de `kza` (leído en sólo lectura: ningún valor nombra `/home/kza/releases`) y sus unidades de serie, las 16 combinaciones pasan. Quedan R5-1 (bajo, previo: un glob de shell sobre los slots dentro de un `sh -c` pasa P14; arreglar antes de la primera compactación real, no antes de construir R1) y R5-2 (informativo: escapes C en plantillas).
- Suite batch en `73e20f3` (`u3b/full-0c/RESULT.md`):
  - Server: verde salvo `test_memory_home_denial[evaluator-False]`, una carrera de la prueba con el mantenimiento de memoria en segundo plano, ajena a 0c y corregida en `a526538` (5/5 en el server; sin el ajuste falla de forma intermitente).
  - Mac: verde tras repetir las tandas cortadas por el límite bajo carga ajena (emulador Android, Docker, VM). En las rehearsals de tiempo, a solas, falló una vez `test_compiled_slot_rollback` con `runtime_admission_writer_active` en la revalidación final: la misma falla intermitente sin causa establecida del 2026-11-06/07 (T024/F1, G01). Pasa a solas en 0c y en el server; 0c no toca `_assert_writers_closed`.
- Los corredores de la suite usan ahora seis tandas en ambas máquinas, con cortes en orden de bytes (`LC_ALL=C`): el `awk` de macOS comparaba por locale y desequilibró la división de tools.
- R5-1 y R5-2 corregidos en la rama `u3b-0c-r51` (sobre `464c08f`, sin integrar ni subir), sólo en `scripts/compaction_units.py`; la regla compartida con la adopción (`adoption_commands.py`) no cambió:
  - R5-1: un comodín de shell (`*`, `?`, `[...]`) en un trozo de ruta absoluto se juzga por la parte fija, componente por componente como lo expande el shell, en unidades normales (`check_value`), plantillas (`_judge`) y entorno del gestor. Rechaza `sh -c 'tar czf /srv/b.tgz <raíz>/slots/*/state'`, `%h/rel*/slots/%i`, `%h/r?leases/slots/*/state`, `%h/*/slots/a`, un `..` tras el comodín y un enlace del home a la raíz (`%h/r?/slots/a` con `~/rl`). Pasan `%h/backups/*.tgz`, `%h/.cache/*`, `/srv/*/data`. En una unidad normal nombrar la raíz misma (`/home/kza/rel*`) sigue pasando, como su ruta; en una plantilla rechaza, como la subcadena. Los comodines relativos (`find -path '*/cache/*'`, `rsync --include='*/'`, el `s/:*$//` de unity) no se juzgan: con la raíz dentro del home habrían rechazado unity.
  - R5-2: las plantillas se leen también como las lee systemd (escapes C decodificados dentro y fuera de comillas, antes de los especificadores; los desconocidos quedan). Comprobado en systemd 255 de kza-yo-lan1 con una plantilla `kza-rehearsal-r51esc@` en el directorio runtime de `yo` (borrada después): `\x65`, `\145`, `\u0065` y `\s` se decodifican; `\d`, `\x00`, `\$`, `\;`, `\x2` quedan; `D\x25h` termina en el home; comillas sin cerrar dejan la unidad `bad-setting`. Se juzgan ambas lecturas y el texto decodificado también decide la tolerancia de U3b-0e4 (sólo en P14).
  - Revisión independiente (`t040-t041/r51-g01-independent-review.json`): «aceptar con cambios», corregidos en la misma rama. A-F2: `[...]` es un carácter de su conjunto y un `[` sin cerrar es literal (antes rechazaban `curl http://[::1]:8200/health`, `awk '/[[:space:]]+/'`, `sed '/[a-z]/d'`). A-F1: un comodín relativo con directorio (`releases/slots/*/state` desde el home) o desde un directorio de trabajo dentro de la raíz se juzga desde ese directorio. A-F3: el guion de un `sh -c` de plantilla se juzga también sin comillas y con `$'...'` decodificado. A-F4: en una unidad tolerada, `$D` sin valor seguido de la cola de un miembro rechaza.
  - Límites aceptados: comodines relativos sin directorio (`*/cache/*`), los `cd` dentro de un guion, expansión de llaves de bash (`rel{e,}ases`) y extglob/zsh.
  - Pruebas y mutantes: ver `r5_fixes` y `r5_review_fixes` en `t040-t041/u3b-0c-evidence.json`. En `c460493`: `test_compaction_units.py` 273 pasan; conjunto de compactación y adopción en el Mac 742 pasan y 26 se omiten; en el server, `test_linux_compaction_units.py` 13 a solas, el conjunto 606 + 136 (5 omitidas) y `test_linux_owner_compaction.py` 1; 38 de 38 mutantes muertos.

- G01 (`runtime_admission_writer_active` en la revalidación final de `test_compiled_slot_rollback`), causa establecida el 2026-11-14 (`t024/g01-writer-active-20261114.json`, rama `g01-writer-active`):
  - Desde 69a79e6 el ensayo corre la app de 80b47f5 (la primera con admisión). Su `src.main` hace su propia admisión, así que el puente de producción de 3fc5ac6 nunca se aplica.
  - En 80b47f5, el mantenimiento histórico de `DiagnosticServices` del candidato abre `DiagnosticStore` sobre el catálogo delegado del slot viejo y retiene `slots/old/state/data/graph/metrics_policy_v1.jsonl.sqlite3.lock` entre `prune()` y `close()`. La revalidación final (`main.py:1943`, sin serializar) sondea ese estado con `_assert_writers_closed` y encuentra el lock del propio proceso. Bajo carga, el `prune` dura más y la ventana crece.
  - Reproducción determinista: una copia descartable de 80b47f5 retiene ese mismo flock 10 s después de `prune()` y falla 3/3 con la traza idéntica. La misma copia pasa 2/2 con el arreglo.
  - Producción no queda expuesta. HEAD ya serializa con `revalidate_constructed` (3fc5ac6) y con `diagnostic_history_validation_gate` (431e44e), y no hay otro dueño de flocks en un estado delegado. La release b0866e8 no tiene admisión.
  - Arreglo, sólo en el ensayo: `managed_slot_graph.serialize_older_history` porta a una app vieja sin compuerta el mutex de 431e44e (trabajador histórico contra la sonda original). La sonda sigue igual y rechaza a cualquier otro escritor.
  - Prueba: `tests/unit/tools/test_managed_slot_history_gate.py`. Los mutantes M1 (mutex nulo) y M2 (sin instalar) quedan muertos.
  - En el server: la copia retenida falla sin el arreglo y pasa con él, y el ensayo normal pasa. Sin la copia, la falla natural es rara: 3/3 corridas sin arreglo pasaron con el Mac bajo carga de CPU.
  - Pasar el ensayo a una revisión con la compuerta no sirve hoy: 431e44e, cdd3c84 y 740ca42 fallan por otras causas con este arnés, y HEAD falla en la copia.

- F4 y F10 de la revisión de los arreglos de Linux, integrados el 2026-11-14 desde la rama `f4-f10-identity`:
  - F4: en Linux, los destinos SQLite nativos se fijan con un descriptor `O_PATH` desde el plan hasta el cierre de la época. Cerrarlo no suelta los locks de SQLite. En macOS no hay pin porque APFS no reutiliza identificadores.
  - F10: los diez temporales propios se borran sólo si el nombre sigue siendo su inodo retenido (`src/core/own_temporary.py`).
  - Pruebas: `tests/unit/memory/test_native_destination_pins.py`, `tests/unit/core/test_own_temporary.py` y tres casos de `test_build_runtime.py`. Evidencia: `t040-t041/f4-f10-evidence.json`. Sección nueva en DEPLOYMENT.
  - Revisión independiente (`t040-t041/f4f5-independent-review.json`, junto con F5 en `batch-f4f5`): «aceptar con cambios». Corregido en `f4-f10-identity`:
    - R1: tras liberar los pins (también después de un cierre con `epoch_incomplete`) no se entrega ningún pin nuevo (`native_pins_closed`, envenena). N9 sigue y es alcanzable: lo prueba un pin histórico liberado.
    - R2: los pins se liberan después de drenar el ejecutor de disco.
    - R6: `EMFILE`/`ENFILE` al abrir un pin da `native_pin_unavailable`.
    - R7: si falla el `dup`, el temporal vacío se borra a través del descriptor de `mkstemp`, en los tres ayudantes y en `diagnostic_store`.
    - R3/R5: huecos conocidos listados en DEPLOYMENT. R9 se resuelve al integrar.
    - Verificado en `d0c22aa`: pruebas nuevas y relacionadas verdes en la Mac y en el server; mutantes 31/31 muertos en el server (N9 incluido) y 23/31 en la Mac, donde los 8 restantes protegen comportamiento sólo de Linux.
- F5 (umask del servicio gestionado) y presupuesto de cierre del coordinador de borrados, 2026-11-14 (integrado el 2026-11-14 desde la rama `f5-umask-close`; evidencia `t040-t041/f5-close-budget-evidence.json`):
  - Consumidores revisados en sólo lectura antes del cambio: ninguna unit, timer ni crontab de `kza` o `yo` lee el estado del slot; `audio_watchdog_alert.py` no corre en ninguna cuenta. El único acceso de otra cuenta es la ACL de `yo` sobre `console.log`, que abre systemd y no el hijo.
  - Cambio: el controlador, `AdmissionController.run` y, por S1 de la revisión, el trabajador de compactación (`compaction.run_worker`) lanzan su hijo con `umask=0o077`, y `managed.conf.template` agrega `UMask=0077`. `audio_health.json` conserva su `fchmod` `0644`. Pruebas con creación real (directorios, archivo, base WAL con `-wal`/`-shm`) bajo umask `0002`; cuatro mutantes muertos (uno por S1). Mac: 524 pasan y 1 se omite en el conjunto del lanzador. Server (`yo`, clon aparte desde bundle): 538 pasan y 1 se omite, con `test_linux_managed_launcher`. Tras S1, `test_stage_runtime_compaction.py` da 38 pasan en las dos máquinas.
  - Presupuesto de cierre, primera medición: la falla de a2b (`test_no_snapshot_without_sources_is_written_or_hydrated`) agotó el presupuesto de **10 ms** que fija el grafo de esa prueba, no el de 1 s de producción. Esos 570 cierres (1-1,5 ms p50, colas de 78-295 ms con el arnés sobresuscrito) miden **sólo el overhead del coordinador**: el grafo offline tiene la memoria apagada y el driver del índice sólo hace un `stat` (S4 de la revisión).
  - S4, segunda medición con el driver real: `PolicyIndexEraser` sobre un Chroma local sintético (vectores aleatorios de 384 dimensiones, 100 a 50.000; nunca datos de producción) y el cierre lanzado con un borrado de 32 ids en curso. Producción tiene ~100 vectores (`memory_db` de 3,7 MB, sólo listado). Con el cliente de `MemoryServices` abierto en la misma ruta («warm», el caso de producción con memoria activa), el cierre no pasa de 24 ms sin carga agregada ni de 0,30 s con el server sobresuscrito ~4x (stress-ng 48 dentro del cupo de 12 CPU de `yo`, con throttling CFS que el slice de `kza` no tiene). Sin otro cliente abierto («cold»: memoria apagada con un índice existente, o un target histórico en otra ruta), bajo esa carga: 0,58 s a 100 vectores, 0,90 s a 10.000, y a 50.000 se pasó de 1 s en 7 de 10 cierres. La Mac estuvo a carga 4,2-6,0 de otros proyectos, sin estrés propio.
  - Si se excede: no hay pérdida de datos. Los claims son durables y se reintentan al arrancar, y la autoridad espera la tarea retenida. Pero no hay recibo: el head queda `unknown` (68) y R-b gasta el cupo de 24 h; un segundo cierre sin recibo da 69, sin reinicio. Es una cota de vivacidad, no de seguridad.
  - Recomendación, decisión del usuario: con la configuración actual (warm, ~100 vectores) 1 s alcanza (~3x de margen bajo carga fuerte). Si importa el camino cold o un índice mucho más grande, subir el presupuesto del coordinador a ~5 s cubre el peor paso medido (1,28 s) con dos pasos seguidos de un target. No se cambió ningún límite; el presupuesto de 0,01 s de la prueba también queda para el usuario.

Siguiente:
- Integradas el 2026-11-14 `u3b-0c-r51` (R5-1, R5-2 y la revisión A-F1..A-F4) y `g01-writer-active` (con B-F1 y la regla de releases anteriores a 431e44e en DEPLOYMENT). Suite batch verde en ambas máquinas sobre R5-1/R5-2 + G01 (`u3b/full-r51-g01/RESULT.md`; las rehearsals de tiempo 8/8 en las dos, incluido `test_compiled_slot_rollback`); los arreglos posteriores de la revisión tocan sólo `scripts/compaction_units.py` y sus pruebas, validados por módulo en la Mac y en el server, con 38/38 mutantes muertos (`u3b-0c-evidence.json`, `r5_review_fixes`). Revisión: `t040-t041/r51-g01-independent-review.json`.
- Integradas el 2026-11-14 F4/F10 (`f4-f10-identity`) y F5 (`f5-umask-close`), con las correcciones de su revisión independiente (`t040-t041/f4f5-independent-review.json`: R1, R2, R6, R7, S1, S4). Suite batch verde en ambas máquinas sobre las ramas antes de esas correcciones (`u3b/full-f4f5/RESULT.md`; rehearsals de tiempo 8/8 en las dos); las correcciones se validaron por módulo en la Mac y en el server, con mutantes muertos (31/31 F4/F10 en el server; S1 en ambas).
- **Decisión del usuario pendiente**: presupuesto de cierre del coordinador de borrados (hoy 1 s). Con la configuración actual alcanza (~3x de margen bajo carga fuerte); en el camino cold con un índice grande bajo carga se excedió (hasta 1,28 s por borrado). Opciones registradas en `t040-t041/f5-close-budget-evidence.json`: dejar 1 s, subir a ~5 s, o que el driver no empiece un borrado nuevo una vez iniciado el cierre.
- Medir el tiempo de compactación una vez sobre una copia del estado de producción antes de la primera corrida real (recomendado por la revisión), y en la ventana de R1 leer `units.templates_scanned` y `units.tolerated_unresolved` del `compact plan` real.

`U3b-0e5` (rama `u3b-0e5`, sobre la cabeza de 0e3): preparación de P7.
- `retire-legacy-state` (Q5) borra el estado congelado de b0866e8 y sus diez copias SQLite del 2026-10-29 (`backups/<63ea8d3>/databases`, Q2 = A), y saca 63ea8d3 y b0866e8 de L4 con un registro inmutable. Se escribe una intención antes del primer borrado, para que un corte se complete repitiendo el comando. Rechaza si L4 se usó desde la adopción.
- `_verify_legacy_slot` resuelve cada enlace del destino de rollback. La usan `uninstall-launcher` y los dos `adoption-abort`.
- DEPLOYMENT: hueco de privacidad acotado a los almacenes mapeados (0d4a F1), sección del retiro Q5 y la época 2 del controlador (0e3 F14).
- **Plazo P7, con responsable**: DÍA 0 es el sello de cierre de W9. L4 vale hasta DÍA 0 + 14 × 24 h, y desde ese momento corre el retiro (Q1 = A: «día 14» son 14 × 24 h). Responsable: el usuario, como operador de `kza`, con la autorización ya registrada (P7 = B y `u3b_0e5_retirement`). Fecha exacta: se anota aquí en W10 (todavía no hay ventana). Después no hay rollback no gestionado, y `/home/kza/kza` no es un destino de rollback (Q3 = A).
- Revisión independiente: la ronda 1 dio «aprobado con cambios»; F3–F7 y F9 se corrigieron y F10 quedó registrado (`t040-t041/u3b-0e5-evidence.json`, `review_round`). La ronda 2 aprobó el 2026-11-13 (`t040-t041/u3b-0e5-independent-review.json`, `round_2`). Faltaba el T-A17 compilado en Linux (U3b-0e4); ya pasó en el server en SC-6a (`/home/yo/projects/runs/sc6a`, `sc6a-review-fix`) y en la suite batch de R1-B (`/home/yo/projects/runs/full-r1b`). Con eso la «Compuerta de programación» está cumplida; lo que sigue faltando antes de la ventana está en el «Siguiente» de SC-6a.

Decisiones del usuario en `t040-t041/user-decisions-20261110.json` (`u3b_*`).

## SC-2 y SC-5a — delegación retirada antes de R1 (2026-11-15)

Decisión del usuario aceptada el 2026-11-15 (`u3b_f8_self_contained_releases`):
releases gestionadas autocontenidas, sin delegación a estados dueños; R1 sale
apenas la delegación esté apagada. Diseño: `t040-t041/sc-release-design-v3.json`
(propuesto; sólo SC-2 y SC-5a, las unidades previas a R1, se implementaron; el
modo release, SC-1, SC-3* y SC-4*, sigue en revisión). Rama `sc2-delegation-off`
sobre `26de262`, sin integrar ni subir. Evidencia:
`t040-t041/sc2-evidence.json`.

- Código (`scripts/stage_runtime.py`, `scripts/build_runtime.py`):
  - Toda concesión (`_authorize_target_delegation`, `delegate`,
    `authorize_target_delegation`, `authorize_derived_state`, y
    `delegate_slot` / `authorize_target_delegation` del módulo) rechaza con
    `runtime_admission_delegation_retired` antes de leer, tomar lock o lease,
    o escribir. El CLI `delegate` imprime `{"refused": ...}` y sale con 78.
  - `activate` de otro slot bajo un head gestionado rechaza con
    `runtime_admission_transition_requires_release_mode` antes de leer el
    candidato; pasan el slot del propio head (`serve`, `run`) y el K de la
    compactación pendiente bajo su compuerta. Sin head, nada cambia.
  - `CONTROLLER_EPOCH` 3 → 4. `SCHEMA_CONTRACT_EPOCH = 1` y
    `schema_contract_document()` (tablas, columnas, markers, conjunto
    protegido, todas las reglas de base del lector como constantes con nombre,
    vocabularios de borrados, participantes, estados de hechos y cambios y
    métodos de dueño, SCv2-P5, desde la revisión SC2-R1 cada enum de `src`
    que una tabla protegida guarda tal cual, y desde AUTHZ-1 los enums de las
    tablas de avance del borrado y atribución de `privacy.db`); `release.json`
    declara `schema_contract`. Esas ampliaciones de vocabularios son una
    enmienda del diseño aceptada el 2026-11-15 como decisión técnica,
    revocable hasta que SC-6a construya R1 (ver «Contrato de esquema
    (época 1)» en DEPLOYMENT).
  - `LEGACY_MANAGED_TRANSITIONS = False`. Sólo los tests lo ponen en `True`
    (fixture `legacy_managed_transitions` y `tests/mocks/legacy_transitions.py`);
    un controlador compilado no puede.
- Tests: `test_sc2_delegation_retired.py` (cada rechazo, compuerta de
  compactación, copia de época 3 con `require_forward_epoch` (26de262)
  bloqueada, cadena existente que valida, sirve
  y se compacta, y raíz de época 3 con dueños concedidos por el CLI compilado
  fijado 26de262 compactada bajo el controlador de época 4) y
  `test_schema_contract_pin.py` (pin del digest, sensibilidad de cada
  constante, vocabularios contra los enums de `src`, inicializadores reales).
  Los mundos heredados (rollback, derivación, cadenas de dueños) se construyen
  bajo el flag y luego se ejercitan sin él; la lista de tests cambiados y su
  motivo está en la evidencia (`changed_tests`).
- Docs (SC-5a): DEPLOYMENT (delegación retirada, sin límite de 8 dueños ni F8,
  época 4 y contrato de esquema, sección «Sin transiciones gestionadas después
  de R1 hasta el modo release»), runbook de la ventana de R1 (época 4 y el
  pendiente de SC-6a) y runbook de compactación (época 4).
- **SC-D5 respondida el 2026-11-15: A** (cómo acotar el hueco sin arreglos de
  código después de R1 hasta el modo release: se acepta el hueco, el modo
  release se empieza enseguida y, ante un problema grave de privacidad, se
  apaga el servicio; el retiro Q5 del día 14 no se posterga). En la misma
  tanda: SC-D2, SC-D4, SC-D6, SC-D7 y SC-D8 = A; SC-D1 y SC-D3 sin objeto en
  el diseño v4. Registro: `u3b_sc_release_mode_questions` en
  `t040-t041/user-decisions-20261110.json`; DEPLOYMENT lo anota en «Sin
  transiciones gestionadas después de R1 hasta el modo release».
- Pendiente antes de la ventana de R1 (SC-6a; hecho el 2026-11-15, ver «SC-6a» más abajo): reconstruir R1 dos veces desde
  el commit integrado y comparar byte a byte; borrar o renombrar todo R1 de
  época 3 y, por AUTHZ3-1, todo archivo, árbol extraído y checkout construido
  desde 69a79e6..b0c322d~1 (sus `stage_runtime` no comprueban la época y
  conceden y activan otro slot en una raíz de época 4), bajo `yo` y en cada
  ubicación de transferencia (listar aquí cada ubicación); fijar el sha256 y el
  commit en el runbook con el paso obligatorio `sha256sum` antes de W1;
  repetir el ensayo de adopción (compilado, ventana real en Linux y T-A17) con
  los rechazos nuevos y el `delegate` + `activate` de una copia de ese rango
  sobre una raíz de época 4.
- Pruebas sobre `cc14903`: suite batch en la Mac verde (12.167 pasan, 47 se
  omiten, 2 xfail; las 8 rehearsals de tiempo a solas pasan; node sin fallas).
  En el server (`yo`, clon temporal desde bundle, ya borrado): los cuatro
  módulos Linux de a uno (7 + 1 + 14 + 13 pasan, con systemd real y unidades
  `kza-rehearsal-*`) y el conjunto SC-2 (625 pasan, 1 omitida, 1 xfail).
  Mutantes: 18 de 18 muertos en la Mac.
- Revisión independiente de SC-2/SC-5a (2026-11-15): 11 hallazgos, 2 medios
  (SC2-R1 vocabularios del contrato, R1 rollback gestionado ofrecido en
  DEPLOYMENT), 5 bajos y 4 informativos. Corregidos con test y mutante muerto:
  contrato ampliado y re-fijado una vez antes de R1 (época 1), reglas del
  lector como constantes (SC2-R2/R5), cada mención de rollback gestionado
  marcada no disponible (R1), `install-controller` época 4 (R2), aviso del día
  14 con SC-D5 (R3), residuo del `controller.json` ilegible (SC2-R3), rechazo
  incondicional en tc4 (SC2-R4/R4) y tres inicializadores más (R7). R6 queda
  para SC-5b. Mutantes de la revisión: 20 de 20 muertos más un control que
  sobrevive al test anterior a propósito. Detalle en la evidencia
  (`review_round_1`).
- Segunda revisión independiente (2026-11-15): 1 hallazgo medio (AUTHZ-1),
  3 bajos (C1-C3) y 5 informativos. AUTHZ-1 se resolvió con la opción A, que
  queda como parte de la misma enmienda (aceptada el 2026-11-15): la regla del contrato pasa
  a ser la relevancia para la privacidad; `ErasureState`, `ErasureError`,
  `IndexCleanupError` y `MemoryOrigin` entran en la época 1, un test
  clasifica cada columna de `erasure_targets`, `erasure_steps`,
  `policy_index_deletions`, `facts` y `fact_sources`, y la época 1 se re-fijó
  por segunda vez antes de R1 (`691c2272…`). DEPLOYMENT agrega a las
  exclusiones `policy_index_deletions.state`, `facts.category` y
  `voice_groups.state`/`reason`. También: W3 del runbook de R1 exige
  `controller_epoch` 4 (C1); DEPLOYMENT ya no dice que `run()` acepta slots
  anteriores (C2) y nombra los otros movimientos del head, `adoption-commit`
  tras un head abortado y el K de la compactación simple, que no conceden nada
  (AUTHZ-2); un test nuevo separa `current` del slot del binding del head
  (C3). Mutantes: 14 de 14 muertos más un control que reproduce el hallazgo
  C3. Suite batch sobre `cf445bf` verde: 12.232 pasan, 47 se omiten, 2 xfail,
  node sin fallas y las 8 rehearsals de tiempo a solas pasan. La primera
  corrida de a2a se cortó en el límite de 1500 s con carga alta y la Mac en
  suspensión, con una falla de tc2 (`sigkill_while_running`); repetida
  partida en dos (a2a1/a2a2, mismo límite) pasó entera. En el server, los
  tres módulos de la revisión pasan (110). Antes de SC-6a hay que repetir la
  suite sobre el commit integrado (C4; hecho, ver «Integración de SC-2»). **Decisiones antes de SC-6a** (resueltas el 2026-11-15 como decisión técnica, ver más abajo): confirmar o revertir la
  enmienda de vocabularios (SC2-R1 + AUTHZ-1; C5 advierte que los 15 de
  SC2-R1 que no son de privacidad vuelven transición de esquema cualquier
  cambio de esos enums) y si `release.json` debe declarar también el sha256
  del contrato (AUTHZ-3, para la revisión de SC-1). Detalle en la evidencia
  (`review_round_2`).
- Tercera revisión independiente (2026-11-15): 1 hallazgo medio (AUTHZ3-1),
  4 bajos (AUTHZ3-2, RV3-1..RV3-3) y 6 informativos. AUTHZ3-1: el bloqueo
  por época sólo cubre las copias de `stage_runtime` que ya tienen
  `require_forward_epoch` (desde b0c322d); toda copia de 69a79e6 a b0c322d~1
  (toda la época 2 y la época 3 anterior a b0c322d) concede y activa otro
  slot en una raíz de época 4. DEPLOYMENT, el runbook de compactación, el de
  R1 y este handoff ya no afirman lo contrario; SC-6a (PS-13) borra o
  renombra todo archivo, árbol extraído y checkout de ese rango y el ensayo
  de adopción suma el caso. La barrera de época 4 que no depende del código
  viejo (congelar las capacidades de delegación existentes y que `serve`
  acepte sólo heads registrados por un controlador de época 4) se difirió a
  SC-1 por decisión técnica del 2026-11-15; hasta entonces rigen la regla de
  operación y la limpieza de SC-6a. AUTHZ3-2: un test clasifica cada columna de texto de
  cada tabla protegida y DEPLOYMENT nombra las columnas de diagnóstico
  (`reason`, `last_error`) y `auth_meta.key`; el pin no cambia. RV3-1 y RV3-2:
  DEPLOYMENT nombra el regreso a H de `compact abort --to-h` y anota que
  `delegate` rechaza siempre con `runtime_admission_delegation_retired`.
  RV3-3: un test corre en proceso la rama del CLI `delegate` posterior a la
  guarda (sólo raíces de prueba heredadas) con el aviso de compactación y su
  falla. AUTHZ3-3 (informativo): con el flag de producción, `activate` ya no
  consulta registros de autoridad derivada para K; sólo la capacidad de su
  compactación lo autoriza. RV3-4: el test de rollback compilado usa
  `usefixtures` y no oculta el ayudante importado. Mutantes: 14 de 14
  muertos. Mac: módulos afectados verdes (164, 6 de integración con autoridad
  derivada y 224 de admisión, serve y multi-hop, con 1 xfail); server (`yo`,
  clon temporal desde bundle, ya borrado): 158 pasan. La suite batch completa
  se repite sobre el commit integrado antes de SC-6a (C4). Detalle en la
  evidencia (`review_round_3`).
- **Integración de SC-2 (2026-11-15)**: merge `dd2d7f8` en
  `recovery/production-2026-10-25` (mismo árbol que `b80ed7f`). Decisiones
  técnicas registradas en la evidencia (`technical_decisions_20261115`): la
  enmienda de vocabularios queda aceptada (revocable hasta que SC-6a construya
  R1) y la barrera estructural de AUTHZ3-1 pasa a SC-1; AUTHZ-3 (sha256 del
  contrato en `release.json`) sigue para la revisión de SC-1. Suite batch
  sobre `b80ed7f` verde (C4): Mac a1 6520 pasan, a2a 1079 (repetida: la
  primera corrida falló por disco lleno y después por una opción del runner,
  `tmp_path_retention_policy=failed`, que reusa nombres de tmp y choca con la
  caché de Chroma; se revirtió la opción, no el código), a2b 1274, a2c 2050,
  b1 1082, b2 235, node sin fallas, 8 rehearsals de tiempo a solas; server
  (`yo`, rama temporal desde bundle, ya borrada y checkout restaurado) siete
  tandas 12.289 casos sin fallas, node y 8 rehearsals. Evidencia en
  `data/autonomous-roadmap-20261102/u3b/full-sc2/RESULT.md`.
- **SC-6a (2026-11-15)**: commits `81ed568` (ensayo y paso del runbook) y el
  commit de documentación que fija el archivo (posterior por fuerza: un archivo
  no contiene su propio hash). Evidencia: `t040-t041/sc6a-evidence.json`.
  - R1 construido bajo `yo` en el server desde `81ed568`, dos veces, idéntico
    byte a byte; la construcción en la Mac también coincide. `release.json`:
    `controller_epoch` 4, `schema_contract` 1, `launcher_contract` 3. sha256
    `3f57bd197f2f1633f32683616abefd3db038466d7a966d9925b24c84fc342a0e`, en `/home/yo/projects/releases-r1/r1-81ed568.tar.gz` (0600,
    directorio 0700; renombrado `*.superseded-r1b` al re-fijar R1 desde
    c6462e5, ver «R1 re-fijado desde c6462e5 tras R1-B»). El runbook de R1 lo fijó hasta
    ese re-fijado; exige `sha256sum "$R1_ARCHIVE"` antes de W1 («Archivo R1
    fijado» y W1-pre). Con R1
    construido, la época 1 del contrato de esquema quedó congelada.
  - Ensayo: `test_sc6a_r1_rehearsal.py` (en proceso: 7 tests, pasan en la Mac
    y en el server; 8 desde la corrección de la revisión) y la ventana real de `test_linux_adoption_window.py` (server, `yo`,
    systemd real: 7 pasan en 261 s, con T-A17 compilado). Agregan (i)–(iv) del
    diseño y AUTHZ3-1: una copia de b0c322d~1 sobre la raíz adoptada de época
    4, después de W9, prepara, concede y activa otro slot (residuo fijado tal
    como es); b0c322d se rechaza. Mutantes en la Mac: 16 de 16 muertos (más 3
    equivalentes por las guardas en capas; detalle en la evidencia).
  - Inventario bajo `yo` (borrado salvo lo indicado):
    - Mac, `~/.claude/jobs/3a8b26c3/tmp/runs/` (e1–e5, m2, stress: archivos
      de época 3 464c08f/5354951/6409dcd y sus árboles) y
      `tmp/mutant-logs/bt-C01..C05` (archivos y clones de 69a79e6..b0c322d~1):
      corridas del roadmap, borradas.
    - Mac, `$TMPDIR/e2e0e3-*` y `exp0e3-*` (R1 de época 1 y 2 con slots y
      controladores extraídos, sondas de la revisión de U3b-0e3) y siete
      `kza-runtime-*` (restos de `build_runtime` cortados, cinco del rango y
      dos de época 3): borrados.
    - Mac, `$TMPDIR/pytest-of-gabriel/pytest-25` (borrado por la retención de
      pytest durante estas corridas) y `pytest-46` en adelante (los de este ensayo):
      borrados.
    - Server, `/tmp/g01-server-runs` (época 3) y
      `/tmp/pytest-of-yo/pytest-102`, `pytest-104` (del rango): borrados.
    - Server, `/home/yo/projects/kza`: estaba en 26de262 (época 3); avanzó por
      fast-forward a la rama integrada.
    - Se conservan (sin copia del rango): el checkout de la Mac, el clon
      `jobs/3a8b26c3/tmp/kza-sc2` (b80ed7f, época 4), `dist/` y `data/` de la
      Mac (releases legadas sin `delegate`), `/tmp/kza-publish-*` del server
      (legadas del 2026-10-29), `/home/yo/projects/runs` (sólo logs) y los
      esqueletos vacíos `/private/tmp/u3b0b` y `u3b1a`. Nada de origen dudoso:
      ningún renombre `*.superseded-epoch3`.
    - Corrección de la revisión (SC6A-R1): el primer inventario buscaba sólo
      por nombre y no vio cuatro copias sueltas renombradas en la Mac, todas
      con `delegate` y sin `require_forward_epoch`: `/private/tmp/sr_head.py`
      y `~/.claude/jobs/3a8b26c3/tmp/sr_head.py` (idénticas, blob de
      389a18c..4bb38e3~1), `jobs/3a8b26c3/tmp/stage_runtime.r.backup.py`
      (69a79e6) y `jobs/3a8b26c3/tmp/stage_runtime.backup.py` (copia de
      trabajo sin commit del 2026-11-10). Restos del roadmap: borrados. El
      inventario se repitió por contenido (todo archivo con `def
      delegate_slot(`, todo `.pyc` que lo nombra y todo miembro de un archivo
      comprimido, clasificados por `require_forward_epoch` y
      `delegation_retired`) sobre el checkout de la Mac, `bench-ambient`,
      `kza-design-reference`, `_otros`, `~/.claude/jobs`, `/private/tmp` y
      `$TMPDIR`; en el server bajo `yo` sobre `/home/yo`, `/tmp`, `/var/tmp`,
      `/run/user/1011` y `/dev/shm`; y del lado de `kza`, sólo leyendo,
      sobre `/home/kza`, `/tmp`, `/var/tmp`, `/run/user/<uid>` y `/dev/shm`.
      Ninguna copia de ese rango ni de época 3 (`jobs/3a8b26c3/tmp/u3b0b.diff`
      es un diff parcial de 515 líneas, no una copia que se pueda correr).
    - Los árboles `/tmp/pytest-of-yo/pytest-6`, `42`, `52` y `103` del server,
      listados arriba como conservados, los borró la limpieza del revisor
      (SC6A-R6); no tenían ninguna copia del rango.
    - `kza` (sólo listado, sin cambios): `/home/kza/releases/slots/*`,
      `incoming/` y `/home/kza/kza` son las releases legadas 2c5adb4,
      b0866e8, 63ea8d3 y e9669ec, sin `delegate`. Nada del rango; para el
      operador sólo queda transferir R1 en la release.
  - Lo que queda: el historial de Git, y la suite vuelve a construir copias
    del rango en el tmp privado de pytest de `yo` en cada corrida (238b463,
    40aa05b, b0c322d~1). No son ubicaciones de transferencia.
- Siguiente: el diseño del modo release (v4 en revisión; las respuestas SC-D
  ya están registradas, ver arriba). La ventana de R1 sigue sin programarse.
  Su «Compuerta de programación» (runbook de R1) está cumplida desde la
  verificación del 2026-11-15: U3b-0e5 tiene su revisión independiente
  (ronda 1 «aprobado con cambios» y ronda 2 aprobada, 2026-11-13,
  `t040-t041/u3b-0e5-independent-review.json`; arreglos en
  `u3b-0e5-evidence.json`, `review_round`) y el T-A17 compilado en Linux
  pasó en el server en SC-6a (`runs/sc6a`, `runs/sc6a-review-fix`) y en la
  suite batch de R1-B sobre el árbol de c6462e5 (`runs/full-r1b`). La
  compuerta no es lo único pendiente antes de la ventana. La re-revisión de
  la ronda de arreglos de SC-6a (cbb509c, 9ec7d95) y la revisión
  independiente de la ronda 1 de arreglos de R1-B1/R1-B2 se hicieron el
  2026-11-15; sus arreglos cambiaron código y R1 se re-fijó desde 357dcb5
  (ver «Re-revisiones de SC-6a y R1-B, y R1 re-fijado desde 357dcb5»). La
  re-revisión independiente de esos arreglos (ronda 2, 2026-11-16) aprobó;
  sus dos hallazgos bajos se corrigieron sin tocar código de ejecución y R1
  sigue fijado desde 357dcb5 (ver «Re-revisión ronda 2»). Las revisiones del
  lado de desarrollo previas a la ventana están hechas. La suite batch completa sobre 51fbc43 pasó en la Mac y en el server (`u3b/full-51fbc43/RESULT.md`); si algo posterior cambia código, R1 se vuelve a construir y a fijar. El visto bueno visual del panel de R1-B2 sigue con
  el usuario. Después quedan los
  pasos del operador: registrar la autorización de la ventana y los ids
  P1=D y L4, transferir a `kza` el R1 fijado (`r1-357dcb5.tar.gz`) y
  verificarlo, el drop-in sudo de W7 y el arranque real hasta READY de W8.
  La ventana la programa el usuario, como operador.

## R5V-1 — dueños de captura huérfanos ya no bloquean parada ni mute (2026-11-15)

Rama `r1b1-orphan-recorder` desde `81ed568`, en un clon del roadmap;
integrada en `recovery/production-2026-10-25` con el merge `c6462e5` y
subida. Evidencia:
`data/autonomous-roadmap-20261102/t040-t041/r1b1-evidence.json`.

- Problema confirmado por la ruta real de parada. Si una pestaña o un cliente
  nativo pierde su token en un reinicio, queda una fila `unknown` que nadie
  puede retirar. Con eso, toda parada gestionada se colgaba sin recibo y
  dejaba el head en `UNKNOWN`. Un mute total además quedaba en `muting` sin
  unmute posible, contra la garantía H0 de reactivar con un control
  autenticado.
- Arreglo técnico (opción e): las filas `unknown` al arrancar quedan como
  *arrastradas*. Drain, `pre_close` y `verify_closed` ya no las esperan;
  `pending()` y `total_mute_confirmed` sí las cuentan. Un titular que
  presenta su token a `poll` o `activate` deja de estar arrastrado y vuelve a
  ser esperado. Para los que nunca contactan, la barrera depende de los
  monitores de cierre seguro de los clientes (nativo 0,25 s/1 s, navegador
  0,5 s/5 s). Es una decisión técnica, no un reconocimiento del usuario: la
  regla del 2026-11-04 («active/unknown owners require explicit stop
  receipt») es un registro técnico; el arreglo conserva «unknown participants
  cannot receive confirmation» y restituye la garantía H0 del usuario (el mute
  siempre se puede reactivar). Sólo la parada y el cierre dejan de esperar a
  esos dueños. La prueba de cierre en
  memoria da `capture_permissions_retired: false` y
  `capture_permissions_carried` (clave del diseño v4); el recibo no los guarda,
  y el cierre sólo deja una advertencia con el conteo en el log. Los dueños
  del proceso actual mantienen la barrera de ACK. La capacidad cuenta sólo
  dueños vivos. Detalle en DEPLOYMENT (R5V-1).
- Tests: el test que reproduce el problema
  (`test_capture_orphan_owner_stop.py`) falla en 81ed568 y pasa con el
  arreglo. Tras la revisión 1, los tests de reinicio de navegador, de los
  cinco tipos nativos y el caso `unknown` de `test_runtime_closure` vuelven a
  su contrato de 81ed568: el titular presenta su token, así que el cierre
  espera su `retire`. Tests nuevos cubren el dueño arrastrado silencioso, el
  que reaparece durante una drain o después de la drain del cierre, dos
  reinicios seguidos y los términos de `pending_live`.
- Revisión 1 (R1B1-1 a R1B1-7), aplicada en la misma rama: reclamo por token
  (`poll`/`activate`), premisa de los monitores documentada, advertencia con
  el conteo al cerrar y clave `capture_permissions_carried`, conteos
  `live_capture_owners`/`carried_capture_owners` en el estado y el panel,
  preflight de dueños vivos en los runbooks de R1 y compactación, reintento
  de `retire` en la página, y redacción corregida sobre subidas y filas sin
  tope. Mutantes: 33 de 33 muertos (30 en la primera corrida; M5, N2 y N8
  después, con tests nuevos). Corridas dirigidas en la evidencia (`review_round_1`).
- Corrección de 5b40447: R1B1-1 no necesita un visto bueno del usuario ni
  bloquea la integración por eso. Es una decisión técnica sobre un registro
  técnico (ver arriba y DEPLOYMENT R5V-1).
- Las filas arrastradas se resuelven con R1-B2 (abajo), según la decisión del
  usuario del 2026-11-15.
- Pendiente, antes de la ventana de R1: la revisión independiente de la
  ronda 1 (R1B1-1 a R1B1-7, junto con la de R1-B2) se hizo el 2026-11-15 y
  aprobó (`r1b1-evidence.json`, `rereview`); su hallazgo RR2-1 se arregló en
  19dd44e. La re-revisión independiente de esa ronda de arreglos aprobó el
  2026-11-16 (ver «Re-revisión ronda 2»); la suite batch completa sobre 51fbc43 pasó en la Mac y en el server (`u3b/full-51fbc43/RESULT.md`).

## R1-B2 — «Olvidar grabaciones perdidas» desde Privacidad (2026-11-15)

Misma rama `r1b1-orphan-recorder`, integrada en `c6462e5`. Decisión del
usuario `r1b1_lost_recordings` (aceptada el 2026-11-15): un administrador
olvida las grabaciones perdidas desde Privacidad, afirmando que no queda
ninguna pestaña grabando; queda un registro sin contenido; hasta entonces el
panel muestra cuántas hay y nunca dice «confirmada». Rechazadas: retiro
automático al arrancar y conservarlas para siempre. Evidencia:
`data/autonomous-roadmap-20261102/t040-t041/r1b1-evidence.json`, clave `r1b2`.

- `POST /api/capture/control/forget-carried` con
  `{"expected_carried": N, "no_capture_running": true}` (estricto). Permiso
  `admin:manage`, el del control de captura global; ningún rol de hogar.
- Olvida sólo dueños arrastrados (desconocidos al arrancar y no reclamados
  por su token), nunca vivos ni reservas. Conteo distinto: 409 sin cambios.
  Un titular que se presenta mientras la transacción hace sus comprobaciones
  la revierte (409). Si lo hace entre la última comprobación y el commit, el
  olvido vale para los demás y su fila se restituye en seguida como dueño
  vivo (R1B2-3). Dos olvidos simultáneos se serializan bajo el lock del
  ledger (R1B2-2). Sin dueños arrastrados es un no-op (200,
  `forgotten_capture_owners: 0`).
- Registro sin contenido sólo en el log (WARNING con el conteo): el contrato
  de esquema de la época 1 está congelado y la auditoría está apagada en R1.
  Sin tablas, columnas ni valores nuevos; el pin del contrato pasa.
- Después, si no queda otro dueño, `total_mute_confirmed` vuelve a ser
  verdadero por las reglas normales y la prueba de cierre da
  `capture_permissions_retired: true`.
- Panel: «Grabaciones perdidas: N», botón «Olvidar grabaciones perdidas»,
  confirmación con casilla y botón «Olvidar N grabaciones perdidas»; un
  conteo nuevo cierra la confirmación. Detalle en DEPLOYMENT (R1-B2).
- Tests: unitarios de `CaptureRecorders.forget_carried`, integración con la
  aplicación real (mute confirmado tras olvidar, dueño vivo intacto, parada
  gestionada con prueba limpia), HTTP (cuerpo estricto, 409, roles 403/401),
  permisos de ruta y node del panel. Resultados y mutantes en la evidencia.
- Revisión 1 (R1B2-1 a R1B2-7, veredicto `changes_needed`), aplicada en la
  misma rama, un commit por hallazgo: tests de integración y HTTP de la
  recomprobación de autoridad dentro de la transacción (mata el mutante X1
  del revisor); olvidos serializados bajo el lock del ledger; restitución del
  titular que vuelve entre la última comprobación y el commit; test de
  `Origin` ajeno y cuerpo de formulario con sesión de operador; test del
  conteo mostrado en el panel (mata X7); el panel encola un cambio pulsado
  durante la consulta periódica en vez de descartarlo. La evidencia ya no
  marca `needs_user_decision` en el nivel superior: la pregunta vieja quedó
  como histórica y reemplazada por `r1b2.wording_correction`. Corridas
  dirigidas en Mac: 225 y 99 pasados (dos procesos), node del panel 13/13 y
  `test_obs_panel.cjs` 23/23. Mutantes: 44 de 44 muertos (los 32 originales,
  B2/B3/B10/B11 re-anclados al código nuevo, más X1, X7 y uno por regla
  nueva). Detalle en `r1b2.review_round_1`.
- Pendiente, antes de la ventana de R1: la revisión independiente de la
  ronda 1 se hizo (aprobada, con RR2-1 bajo y RR2-2 informativo; ver abajo)
  y la re-revisión independiente de su ronda de arreglos aprobó el
  2026-11-16 (ver «Re-revisión ronda 2»); el visto bueno visual del panel
  sigue con el usuario como en las demás pantallas.

## R1 re-fijado desde c6462e5 tras R1-B (2026-11-15)

R1-B1/R1-B2 se integraron en `recovery/production-2026-10-25` con el merge
`c6462e5` (mismo árbol que `a8daf68`, el de la suite batch de R1-B). Cambian
`src/` y el frontend, así que el R1 fijado por SC-6a (81ed568) ya no era el
código a instalar. Evidencia: `t040-t041/sc6a-evidence.json`, clave
`r1b_repin`.

- R1 construido bajo `yo` en el server (`/home/yo/projects/kza`, avanzado por
  fast-forward a `c6462e5`) con `scripts/build_runtime.py --revision
  c6462e5f2794e21e797f6f8b397f1f3bf911dd29`, dos veces: `cmp` idéntico byte a
  byte; una tercera construcción en la Mac (CPython 3.13.18 contra 3.13.12)
  también da los mismos bytes. `release.json`: `controller_epoch` 4,
  `schema_contract` 1, `launcher_contract` 3; 416 módulos compilados, 476
  archivos; `stage_runtime.pyc` con `delegation_retired` y
  `require_forward_epoch`. Frente al de 81ed568 cambian sólo `release.json` y
  los siete archivos de R1-B.
- sha256 `733b6fc35dc929accc00b78b231092531a7ca131663ccc8b663f9d9e48851b86`,
  en `/home/yo/projects/releases-r1/r1-c6462e5.tar.gz` (0600, directorio
  0700). El anterior quedó al lado, sin borrar, como
  `r1-81ed568.tar.gz.superseded-r1b` (sha256 `3f57bd19…`); `SHA256SUMS`
  lista los dos con sus nombres nuevos. La copia de la Mac y la segunda del
  server se borraron; `build-c6462e5-a/b/build.json` quedan como registro.
- El runbook de R1 fijó `R1_SHA256` y `R1_COMMIT` de c6462e5 (commit de
  documentación posterior, por fuerza) hasta el re-fijado desde 357dcb5
  (abajo); ese archivo quedó renombrado
  `r1-c6462e5.tar.gz.superseded-rr2-1`. `test_the_runbook_pin_is_the_archive_of_its_commit`
  reconstruye desde el commit fijado y da el sha256 fijado.
- Nada cambió del lado de `kza`; sin release ni instalación.

## Re-revisiones de SC-6a y R1-B, y R1 re-fijado desde 357dcb5 (2026-11-15)

Las dos rondas de arreglos que habían salido sin re-revisión (SC-6a:
cbb509c, 9ec7d95; R1-B1/R1-B2 ronda 1: 7e1dffa..5b40447 y 7864310..b7f9558,
en `c6462e5`) se revisaron de forma independiente. Evidencia:
`t040-t041/sc6a-evidence.json` y `t040-t041/r1b1-evidence.json`, clave
`rereview` en los dos.

- SC-6a, veredicto `changes_needed`:
  - RR-1 (media): el día 14 corría `$ST`, que sólo existe en la sesión de
    la ventana. El runbook y DEPLOYMENT definen ahora `PY`, `C` y `ST` desde
    `controller/current` al empezar el día 14; T-A17 corre los dos comandos
    del día 14 con el controlador instalado (357dcb5).
  - RR-2 (baja): la recuperación tras perder la sesión no definía `$PY` y
    daba un DETENER falso. El bloque de extracción define `PY` y las dos
    notas de recuperación nombran el bloque `PY=`/`C=`; el ensayo ya no le
    agrega `PY` a los bloques (357dcb5).
  - RR-3 (baja): esta entrada de R5V-1 no tenía «Pendiente» y
    `r1b1-evidence.json` decía `integrated: false`. Corregido (este commit de
    documentación).
  - RR-4 (info): el ensayo de extracción fallaba en la Mac con otro
    `TMPDIR`; ahora toma el directorio de un `mktemp -d` de prueba, y la
    frase de W1-pre sobre `$ST` queda fijada por un test (357dcb5).
  - RR-0: lo demás se verificó sin defectos.
- R1-B1/R1-B2, veredicto `approve`:
  - RR2-1 (baja): un olvido que ya estaba en su transacción cuando empezaba
    el `aclose` podía olvidar a un titular que presentó su token en ese
    momento, y la prueba de cierre daba retirado. Arreglo (19dd44e): el
    olvido se rechaza (503) desde que empezó el cierre; si el cierre empieza
    entre la última comprobación y el commit, restituye todas las filas. Un
    rechazo ya no hace fallar el `aclose`. b02116c corrige una frase de
    DEPLOYMENT que rompía `test_sc5a_docs`. Mutantes: 4 de 4 muertos.
  - RR2-2 (info): la evidencia de R1B2-2 describía el mutante sin lock como
    antes de R1B2-3. Corregida la evidencia y DEPLOYMENT: con la
    restitución, el lock es una segunda defensa.
- R1 re-fijado: RR2-1 cambia `src/memory/capture_recorders.py`. R1 se
  construyó bajo `yo` en el server (`/home/yo/projects/kza`, avanzado por
  fast-forward a `357dcb5`) con `scripts/build_runtime.py --revision
  357dcb57816834a9f4bfffb9bb5357ebd13d926e`, dos veces, con `cmp` idéntico, y
  una tercera vez en la Mac, con los mismos bytes. `release.json`:
  `controller_epoch` 4, `schema_contract` 1, `launcher_contract` 3; 416
  módulos compilados y 476 archivos. Frente al de c6462e5 cambian sólo
  `release.json` y `src/memory/capture_recorders.pyc`. sha256
  `c5305d588a228aca445489b90c00a2b11165d24f8fa1078ad19a68fbe9630a24`, en
  `/home/yo/projects/releases-r1/r1-357dcb5.tar.gz` (0600, directorio 0700).
  El de c6462e5 quedó al lado, sin borrar, como
  `r1-c6462e5.tar.gz.superseded-rr2-1`; `SHA256SUMS` lista los tres archivos
  y `sha256sum -c` da OK. La copia de la Mac y la segunda del server se
  borraron; `build-357dcb5-a/b/build.json` quedan como registro.
- Corridas: Mac, dirigidas de captura 278 pasados y SC-6a (docs y ensayo) 30
  pasados, más el ensayo de extracción con un `TMPDIR` privado. Server
  (`yo`, sobre 357dcb5, de a una): `test_linux_adoption_window` 7 de 7 en
  262 s y 153 dirigidas en proceso (`/home/yo/projects/runs/rr-fixes`).
- Pendiente antes de la ventana (actualizado el 2026-11-16, ver «Re-revisión
  ronda 2»): la re-revisión independiente de esta ronda de arreglos aprobó.
  La suite batch completa sobre 51fbc43 pasó en la Mac y en el server (`u3b/full-51fbc43/RESULT.md`); queda el visto bueno visual del
  panel de R1-B2. Nada cambió del lado de `kza`; sin release ni instalación.
  La ventana sigue sin autorizar ni programar.

## Re-revisión ronda 2 (2026-11-16)

Re-revisión independiente de la ronda de arreglos 2139fa3..51fbc43 (RR-1 a
RR-4, RR2-1 y RR2-2). Evidencia: clave `rereview2` en
`t040-t041/r1b1-evidence.json` y `t040-t041/sc6a-evidence.json`. Veredicto:
`approve`, con dos hallazgos bajos y uno informativo.

- RR3-1 (baja, corregido): el mutante M5 (`_forget_carried` captura sólo
  `PermissionError`, no `ValueError`) sobrevivía a todos los tests de
  captura; la nota M3 de la evidencia decía que cubría `ValueError`. Test
  nuevo `test_a_count_conflict_drained_by_close_does_not_fail_it`: el
  SELECT de los arrastrados queda retenido, empieza el `aclose`, y un
  conteo equivocado (1 frente a 2) da `capture_carried_count_conflict` sin
  hacer fallar el cierre. M5 muere (`aclose` da
  `capture_recorder_state_unconfirmed`). Nota M3 corregida.
- RR3-2 (baja, corregido): el día 14 decía «Desde W3 sólo se corre
  `controller/current`», pero W7 y W-abort-2 todavía corren el `$ST`
  extraído. La frase dice ahora que ese `$ST` no sobrevive a la sesión de la
  ventana y que el día 14, como W4 y `completion-status`, corre desde
  `controller/current`; `test_sc6a_docs` la fija. Los comandos de W7 y
  W-abort-2 no cambiaron.
- RR3-3 (info, diferido al backlog posterior a R1): endurecimiento opcional
  en `src` (excepciones después del commit en `_forget_transaction` como
  falla más `RuntimeError`, o `_read_cache` con `RuntimeError` ante un
  estado desconocido). No bloquea la ventana.
- RR3-0: lo demás se verificó sin defectos.

Sin cambios en `src/` ni `scripts/`: R1 sigue fijado desde 357dcb5 (sha256
`c5305d588a228aca445489b90c00a2b11165d24f8fa1078ad19a68fbe9630a24`). Las
revisiones del lado de desarrollo previas a la ventana están hechas. La suite batch completa sobre 51fbc43 pasó en la Mac y en el server (`u3b/full-51fbc43/RESULT.md`). Queda el visto bueno visual
del panel de R1-B2 en Privacidad, que sigue con el usuario, y después los
pasos del operador. La ventana la programa el usuario; sigue sin autorizar ni
programar.

## SC-1a — controlador de época 5 e instalación T0 (2026-11-16)

Primera unidad del modo release posterior a R1 (diseño
`t040-t041/sc-release-design-v4.json`, unidad SC-1a, aceptado). Rama
`sc1a-controller-epoch5` desde b15ce86, sin push; la suite batch y el merge
los hace la sesión coordinadora. Sin release: producción y R1 siguen en
época 4. Evidencia: `t040-t041/sc1a-evidence.json`.

- `CONTROLLER_EPOCH` 5 y `CONTROLLER_REVISION` 1; `build_runtime` registra en
  `release.json` la clausura de imports del controlador
  (`controller_sources`, unas 300 fuentes) y su digest
  (`controller_sources_sha256`). El triple instalado se lee de
  `controller/current/release.json`.
- `install-controller`: monotonía en (época, revisión), mismo par con otro
  digest rechazado; primera instalación de época 5 con log pendiente o
  ilegible, head abierto, registros de delegación o dueños rechazados, y SR-1
  (binding recalculado en proceso, sonda del slot, contrato de esquema,
  comprobaciones de slot de `serve`); siguientes: SR-1 con head cerrado, head
  abierto sólo bajo un log del modo release con el binding de H igual,
  head desconocido rechazado. Un T0 compilado debe correr desde el archivo
  nuevo extraído.
- CS2-9: `runtime_admission_controller_install_incomplete` fuera de la
  compuerta `install` (fila T0S).
- Guarda AUTHZ3-1 sin estado (`runtime_admission_delegation_frozen`,
  `runtime_admission_owners_present`,
  `runtime_admission_compaction_owners_present`); L4 no queda bloqueado.
- Tests: los mundos con dueños corren con una construcción de época 4 del
  mismo árbol (`tests/mocks/epoch4.py`, fixtures `epoch4_controller` y
  `x4_archive`); el ensayo in-process de R1 (SC-6a) también, porque R1 es
  época 4. Los rechazos de época 5 están en
  `tests/unit/tools/test_sc1a_controller_install.py`.
- Pruebas dirigidas: Mac, módulos unitarios de admisión, compactación,
  lanzador, contrato de esquema y docs de SC-5a/SC-6a, más los ensayos de
  integración con dueños (compilados, rollbacks, reconciliación, checkpoints
  a solas), sin fallas tras las correcciones; server (`yo`, rama temporal
  desde bundle, checkout restaurado en b15ce86) los cuatro módulos Linux de
  systemd, SC-1a y el ensayo de SC-6a, sin fallas. Mutantes 31/31.
- Queda para SC-1b: escribir `intent.mode = "release"` (la instalación ya lo
  lee), P6'(c) con el triple (plan sin T0 rechazado), `t0_required` y
  `controller_behind`, y el resto de plan/start del modo release. El test
  completo de CR2-3 (V2, controlador corregido y `abort --to-h`) necesita
  SC-4a/SC-4b.

### SC-1a — revisión 1 aplicada (2026-11-16)

Dos revisiones independientes pidieron cambios; aplicados en la misma rama
(sin push). Diseño sin cambios y sin decisiones nuevas del usuario; las
lecturas más estrictas quedan como notas de diseño en `sc1a-evidence.json`
(`review_round_1`, notas SC1A-N5..N17).

- La instalación compara el triple con `controller/current/release.json`
  siempre que exista `controller/current`, aunque `controller.json` falte o
  esté dañado; un manifiesto ilegible rechaza (`controller_tampered`) y sólo
  se repara a mano.
- Instalaciones siguientes: leen el log por su cuenta (ilegible rechaza);
  con head cerrado un log simple pendiente rechaza
  (`compaction_pending`); un head abierto tiene que ser el H o el K
  autorizado del log del modo release.
- Identidad del instalador: toda la clausura del árbol que corre la
  instalación tiene que ser la del archivo (los `.pyc` compilados, o los
  `.py` contra `controller_sources` desde un checkout).
- `_read_delegated_observation` corre la guarda AUTHZ3-1 antes de leer.
- Brecha documentada (sin salida): un T0 con el `$ST` de época 4 sobre una
  compactación de dueños pendiente deja el servicio caído sin salida por
  herramienta. Precondición nueva del T0: ninguna compactación de dueños
  pendiente (se retira o se aborta en la época 4; un controlador corregido
  para terminarla es de época 4). DEPLOYMENT y el runbook de compactación
  (paso 3) corregidos.
- Tests nuevos: guarda en instalaciones siguientes, CS2-9 con manifiesto
  ilegible, `managed_launcher.pyc` ajeno, instalador desde fuentes, SR-1
  paso 4 (`check_slot`), proyección cambiada, T0S de una instalación
  siguiente, la parada terminal anterior y R1 real (357dcb5): su propio
  código calcula el mismo binding, el T0 pasa SR-1 y su slot sirve en
  época 5.
- Pruebas dirigidas tras la revisión: Mac, módulos unitarios (SC-1a, lanzador,
  admisión, compactación y sus revisiones, contrato de esquema, docs de
  SC-5a/SC-6a) e integración (adopción, retiro de estado, ensayo de SC-6a,
  lanzador compilado, compactación de dueños, cadenas, compatibilidad de
  registro de salida, rollbacks), sin fallas; server (`yo`, rama temporal
  desde bundle, checkout restaurado en b15ce86) SC-1a, los cuatro módulos
  Linux y SC-6a, sin fallas. Mutantes de la revisión 16/16 (total 47/47).

## SC-1b — plan y start del modo release (2026-11-16)

Segunda unidad del modo release posterior a R1 (diseño
`t040-t041/sc-release-design-v4.json`, unidad SC-1b, aceptado). Rama
`sc1b-release-plan-start` desde 748df3f, sin push; la suite batch y el merge
los hace la sesión coordinadora. Sin release. Evidencia:
`t040-t041/sc1b-evidence.json` (notas de diseño SC1B-N1.., lista
`follow_up`).

- En época 5 todo log nuevo es del modo release: `intent.mode = "release"`
  (lo lee la instalación de SC-1a), `--retire-prepare-leftovers` forzado, una
  entrada no probada bajo `slots/` rechaza salvo los dos slots heredados del
  registro de adopción confirmado.
- P6' en `compact plan`: contrato y formato del NUEVO, época del NUEVO entre
  5 y la instalada (R1 nunca es destino), served(root) o el mismo triple;
  si no, `t0_required`, `controller_behind` o `controller_revision_conflict`
  con los dos triples; H pasa las comprobaciones de slot; SR-2 (mismo
  contrato de esquema en NUEVO, controlador instalado y H). El digest liga
  modo y archivo NUEVO.
- `compact start`: K desde el archivo NUEVO (`_k_runtime` contra el
  intent), A1 con `schemas` de K iguales a los de H.
- Hasta SC-4a/SC-4b: `compact verify` de un log del modo release rechaza
  (`release_verify_unavailable`); `allow-run`, `--discard-k-writes` y Q6
  rechazan (`plain_mode_only`). La salida es la abort simple antes de la
  corrida (fila A0).
- Reubicación de `pending_audio_deletions.path` y `audio_write_intents.path`
  en toda compactación (antes una cola no vacía bloqueaba), con la misma
  reubicación en la proyección esperada de A1 y colisiones rechazadas.
- Tests: `tests/unit/tools/test_sc1b_release_plan.py`; el test de SC-1a que
  planificaba con el archivo de época 4 después del T0 ahora planifica con
  X5 y comprueba que el de época 4 rechaza.
- Pruebas dirigidas (server, `yo`, clones propios en `runs/sc1b` y
  `runs/sc1b-mut`, sin tocar `kza`): SC-1b, SC-1a, compactación y sus
  revisiones, SC-2, unidades, admisión, lanzador, contrato de esquema,
  `build_runtime`, `owner_compaction`, docs de SC-5a/SC-6a, `tests/unit/core`
  e integración (compactación compilada y Linux, ensayo de SC-6a, adopción,
  lanzador compilado y Linux), sin fallas tras corregir tres tests de SC-1a.
  Los ensayos compilados con presupuesto de tiempo (checkpoints, reconciliación,
  copias) no se corrieron: no tocan lo que cambió. Mutantes 34/34.
- Queda para después (`follow_up`): `--accept-regressed-source` (R5-1, con el
  testigo de recuperación de SC-4b).
- Revisión 1 (dos revisores, `changes_needed`), aplicada en commits propios:
  - RA-1/B1: `compact resume` no repite P6'. `_k_runtime` compara ahora el
    `release.json` del archivo dado con el del intent **antes** de escribir
    nada; un `resume` con el archivo de H (o un NUEVO no admitido) después de
    un `start` cortado rechaza `compaction_artifact` con `"archive":
    "not_the_intent_archive"` y el `resume` con el NUEVO sigue llegando a la
    activación. P6'(f) queda probado con dos cambios entre la comprobación y
    la extracción (otro archivo; `release.json` de K reescrito).
  - RA-2: modo simple es sólo un log **sin** `mode`; cualquier otro valor es
    `compaction_log_invalid` (lanzador, verify, allow-run, descarte, Q6,
    served(root)); `status` lo lista como inválido; el abort simple antes de
    la corrida sigue.
  - B3: un NUEVO de época mayor que la instalada rechaza `t0_required` con los
    dos triples (antes `release_epoch`); menor que 5 sigue `release_epoch`.
  - B4: `start`/`resume` de un log del modo release dicen `next: compact
    abort <id>` hasta SC-4a; un log simple sigue con `compact verify`.
  - B2/B5: runbook de compactación con `NEW_ARCHIVE` (época 5: `--archive` es
    siempre el NUEVO, también en `resume`) y filas con salida para
    `t0_required` por época, `controller_revision_conflict`,
    `release_unproven_copy` (un `slots/.prepare-*` vacío se quita con `rmdir`;
    otra ruta: servicio en H, investigar, decisión de 48 horas),
    `compaction_artifact`, `schema_changed` y `locator_collision`.
  - RA-3/RA-4/RA-5: tests de la exención heredada (slot de estado heredado,
    dos registros confirmados, nombre de registro distinto del digest), de
    las tres patas de SR-2 y de las colisiones de cola en las dos tablas.
  - RA-6, RA-7 y B6 van a `follow_up` (no bloquean la primera actualización
    posterior a R1).
  - Pruebas (server, `yo`, `runs/sc1b` y `runs/sc1b-mut`, salidas en
    `runs/sc1b-r1-out/`): SC-1b 15/15, SC-1a 45/45, módulos de compactación
    507 pasan y 5 saltados, admisión y lanzador, integración de compactación
    compilada y Linux y el ensayo de SC-6a, y el control final (docs de
    SC-5a/SC-6a, SC-1b, `tests/unit/core`) 541/541. Dos fallas intermedias se
    corrigieron: el test de SC-1a que leía `mode: "plain"` como simple (ahora
    `log_invalid`) y un `next` agregado al no-op de `resume` que el ensayo
    compilado fija (revertido). Mutantes de la revisión 17/17.

## SC-4a — corrida de verificación aislada del modo release (2026-11-16)

Tercera unidad del modo release (diseño `t040-t041/sc-release-design-v4.json`,
unidad SC-4a, `mechanism.v_verification_run`, enmiendas A1/A2). Rama
`sc4a-verification-boot` desde `batch-sc1b` (561e80e), sin push; la suite
batch y el merge los hace la sesión coordinadora. Sin release. Evidencia:
`t040-t041/sc4a-evidence.json` (notas de diseño SC4A-N1.., `follow_up`,
mutantes, CR-7).

- Se quita la compuerta SC1B-N3 (`release_verify_unavailable`), y la
  revisión ronda 1 (RA-1/RB-1) la reemplaza por una compuerta de release
  hasta SC-4b (SC4A-N13): mientras `RELEASE_EXITS_AVAILABLE` sea `False`,
  `compact verify` de un log del modo release rechaza
  `release_exit_unavailable` y `start`/`resume` dicen `next: compact abort
  <id>` (fila A0, sin pérdida). SC-4b la levanta junto con las salidas de V.
- `serve` agrega `KZA_RUNTIME_RELEASE_VERIFY=1` sólo a la corrida armada de un
  log del modo release, después del filtro de `KZA_RUNTIME_*`. El hijo la toma
  una vez (`take_release_verify_flag`), construye el grafo con
  `verify_only` (lista `skip_list` completa, autoridad del tablero con
  ROLLBACK) y `Application.run` toma la rama de verificación: prepara el
  pipeline, envía `VERIFY=<run_id>` y `READY=1`, y espera la parada.
- `serve` escribe `admission/runs/<run_id>.ready.json` (sólo con el
  reconocimiento antes de `READY`) y `<run_id>.smoke-stop.json` (N segundos
  monotónicos, motivo), detiene al hijo y sale 0. `verified` del modo release
  exige los dos registros y `measured_seconds >= N`. `compact status` muestra
  `verify_branch_reached`.
- Cierre de captura en V: sin borrar reservadas ni esperar, filas iguales a
  las del arranque, cantidad llevada en la prueba de cierre.
- SC4A-N1: hasta SC-4b, `compact retire` de un log del modo release rechaza
  `release_retire_unavailable`; con la compuerta SC4A-N13 ningún log del
  modo release llega a V antes de SC-4b.
- SC4A-N14 (RB-2): N del modo release explícito, de 10 a 60
  (`release_smoke_seconds`); con los valores de hoy, 10 s en la primera
  transición (con T0). El modo simple mantiene 120 s por omisión.
- CR-7: `tools/sc0/run_states.py --verify --revision <commit>
  --template-revision <commit de H>` corre los estados de SC-0 más
  `no_default_control_row` con la lista activa, sobre la plantilla de H del
  commit dado (RA-4: H de R1, 357dcb5).
- Tests: `tests/unit/tools/test_sc4a_verify_run.py`,
  `tests/integration/test_sc4a_verify_boot.py`,
  `tests/unit/memory/test_sc4a_verify_stores.py`,
  `tests/unit/core/test_sc4a_application_verify.py`; mutantes con
  `tools/sc4a_mutants.py`. Resultados en la evidencia.

### SC-4a — revisión ronda 1 (2026-11-16)

Dos revisiones independientes (A y B) pidieron cambios. Registro completo en
`t040-t041/sc4a-evidence.json` (`review_round_1`).

- RA-1/RB-1 (media): con SC1B-N3 quitada, armar un log del modo release era
  un estado alcanzable por código sin salida antes de SC-4b. Ahora es una
  compuerta en el código (SC4A-N13): `RELEASE_EXITS_AVAILABLE = False` en
  `scripts/compaction.py`; `compact verify` rechaza
  `release_exit_unavailable` y `start`/`resume` dicen `next: compact abort
  <id>`. SC-4b la levanta. DEPLOYMENT y el runbook la registran como
  compuerta de release.
- RB-2: N del modo release obligatorio y de 10 a 60 (SC4A-N14,
  `release_smoke_seconds`); el runbook da la fórmula y los valores de hoy.
- RB-3: `compact status` muestra `verify_branch_reached: unreadable` y la
  fase cuando un registro no se puede leer, en vez de `invalid`.
- RA-2: tests nuevos para un registro con el `run_id` de otra corrida y para
  una fila de grabadora cambiada sólo en memoria. RA-3: la traza exime por
  sitio fijado `(tabla, verbo, archivo:función)`. RB-5: un mutante cuenta como
  matado sólo con salida 1.
- RA-4: CR-7 con H de R1 (plantilla construida desde 357dcb5) y V de SC-4a
  (17722da): los 24 estados de SC-0 sellan, VERIFY antes de READY, intervalo
  sostenido y diferencia de proyección vacía; la única escritura proyectada
  fuera de la autoridad es la exenta `ambient/control.py:98`;
  `no_default_control_row` difiere sólo en `ambient_runtime_control`
  (SC4A-N12). Resultados en `t040-t041/sc4a-cr7-r1h-results.json`.
- RA-5: DEPLOYMENT y el runbook dicen que «aislada» es sin entrada ni efectos
  en la casa, no sin red (mismas pruebas de conexión a HA y al gateway LLM).
- RA-6, RB-4, RB-6: sin cambio de código; pasan a `follow_up` para SC-4b y
  SC-R.
- Server (yo, `runs/sc4a2`, con nice y la compuerta pgrep): módulos dirigidos
  más sc1b 72 passed; `tests/unit/core` más sc0/sc1a/sc5a/sc6a docs 577
  passed; módulos de compactación más sc1b y SC-4a 561 passed, 5 skipped.
  Mutantes 57/57 matados, todos con salida 1 (`runs/sc4a-mut`).

## SC-4b — salidas de V: W, abort y retiro del modo release (2026-11-16)

Cuarta unidad del modo release (diseño `t040-t041/sc-release-design-v4.json`,
unidad SC-4b: `mechanism.w_abort_predicate`, `retire`, `safe_stop`, filas de
`crash_table`). Rama `sc4b-w-retire-abort` desde `batch-b` (f9f1d03), sin push;
la suite batch, la revisión independiente (SC-R) y el merge los hace la sesión
coordinadora. Sin release. Evidencia: `t040-t041/sc4b-evidence.json` (notas
SC4B-N1.., `follow_up`, pruebas por módulo, mutantes, preguntas abiertas).
Prioridad del usuario: sólo lo que hace segura la primera actualización
posterior a R1; lo demás queda en `follow_up`.

- W0 en `scripts/compaction.py`: sin hijo vivo de V, escritores de K'
  cerrados, H intacto. Transitorios (no escriben nada): `writer_active`,
  `busy`, `sdk_version_unavailable`, `compaction_sdk_changed` (el binding de
  H difiere sólo en `sdk_versions`) y `compaction_h_transient` (errno y
  códigos SQLite de la lista, por código). Todo lo demás:
  `compaction_h_changed` con el código de abajo (HD).
- W1: K' contra la proyección autorizada de K, abriendo cada base con
  `mode=rw` (recupera journal caliente o WAL en el lugar, nunca crea), con la
  selección compartida (`stage_runtime._authority_selection`, digest dorado);
  informe sin contenido con estado por base y `verify_branch_reached`, sin
  reloj. Clases `equal` / `loss_only` / `restrictive_or_unclear` con
  `W_LOSS`/`W_GRANT` cerradas.
- `compact abort --to-h` del modo release usa W (Q6 queda para el modo
  simple): `equal`/`loss_only` vuelven a H; `restrictive_or_unclear` es la
  parada segura SS con el informe y su digest; `--discard-k-changes <digest>`
  es su única salida (SC-D8 A), registrada en `abort_started` o, después de
  un head movido, en `abort_rechecked` (R5-OPS-1).
- `compact retire` del modo release: binding de K' (regla de
  `sdk_versions`), prueba de cierre, W0 y W1 `equal`, autoridad completa de
  K' igual a la autorizada (SC4B-N4) y escaneo de resurrección absoluto
  (SC4B-N5). Nada escrito en un rechazo salvo el `verified` de siempre.
- `adopt` y `adopt --dry-run` rechazan `compaction_pending` con un log del
  modo release pendiente (R5-PRIV-3). `compact status` muestra `armed_run`
  (`not armed` / `not consumed` / `consumed`) y `verified`, sin W.
- `RELEASE_EXITS_AVAILABLE = True` (se levanta SC4A-N13); se quitó la negativa
  `release_retire_unavailable` (SC4A-N1). Las pruebas de SC-1b y SC-4a ahora
  esperan `next: compact verify` y el retiro completo; una prueba fija la
  compuerta con la constante en `False`.
- Quedan en `follow_up` (sólo después de la aceptación o refinamientos):
  `closed_witness` por registro de recuperación y `--accept-regressed-source`
  (fila PA; hasta entonces su salida es L4 mientras exista), la línea base
  salada de resurrección de K con el reencolado del residuo no protegido, la
  sombra de `adopt --dry-run` bajo `admission/dry-run/`, el campo
  `capture_permissions_carried` del recibo y el `smoke_elapsed` exigido en el
  registro de salida (RA-6 de SC-4a). Corrección de SC-5b (revisión RR3): PA
  y la sombra del dry-run (fila DR) no son sólo posteriores a la aceptación;
  sin log pendiente el dry-run corre también en la primera transición, y las
  dos filas se implementan antes del día 14 de R1 (SC5B-Q1 y SC5B-Q2).
- Tests: `tests/unit/tools/test_sc4b_w.py` (clases, 16 casos por tabla y
  dirección, ledger, bases faltantes o ilegibles, valor sin clasificar,
  journal caliente, mapeo W0, digest dorado) y
  `tests/unit/tools/test_sc4b_release_exits.py` (filas A0, V1-V4, AT, AB, SS,
  HD, RS, RX, C, N12 y el descarte sobre un log simple). Mutantes:
  `tools/sc4b_mutants.py`. Resultados del server en la evidencia.

### SC-4b, revisión 1 (2026-11-16)

Dos revisiones independientes (A: privacidad y autoridad; B: filas del
`crash_table` y operación) pidieron cambios; se aplicaron en la misma rama
(`review_round_1` en la evidencia, con las corridas del server y los
mutantes).

- RA4B-1: W1 ya no lanza `TypeError`/`AttributeError` con un valor de otro
  tipo en K' (un `sources_json` nulo, un `expires_at` de texto, un usuario
  que no es objeto): cuenta como fila sin clasificar o cuenta
  `unclassifiable`, y cualquier otra falla al leer una base de K' es
  `unreadable` con un código sin contenido. Así SS siempre tiene su informe
  y su digest.
- RA4B-4: W1 abre una base, cuentas o configuración de K' sólo si es un
  archivo regular en su propia ruta resuelta, antes del `mode=rw`: un
  directorio de K' enlazado a H ya no le hace checkpoint al WAL de H.
- RA4B-5/RB-7: un lock SQLite de otro proceso sobre una base de K' (o un
  errno de recursos) es el transitorio `compaction_k_transient`, no
  `unreadable`; EIO y el daño siguen en SS (SC4B-N19).
- RB-1: una entrada de lock de K' que no se puede abrir (enlace simbólico,
  sin permiso) ya no deja el abort sin salida: W0 la informa como
  `k_writers: unverifiable` y la clase es `restrictive_or_unclear` (SS y su
  descarte); el borrado de K después de volver a H rechaza el enlace sin
  seguirlo, se quita a mano y `compact resume` termina.
- RB-3: una prueba de cierre de K' ilegible después de V es
  `compaction_k_closure_unreadable` (abort y retiro), distinta de
  `head_moved`; ambas son paradas como HD documentadas en el runbook.
- RA4B-2, RA4B-3, RB-2: pruebas de punta a punta de cada disparador de HD
  (escritura de una copia vieja, tabla borrada, base sobrescrita, marcador,
  prueba de cierre malformada), de `--confirm-h`, del descarte sin nada que
  descartar (`equal` y `loss_only`), de `head_moved`, de
  `k_binding_changed` seguido del abort y del hijo vivo de V; el chequeo del
  hijo vivo quedó sólo en W0 (SC4B-N21).
- RA4B-6, RB-6, RB-5: runbook (re-comprobación con un informe nuevo pide
  repetir los pasos de SS; viñetas viejas marcadas «modo simple»; anotar
  `source_binding_id` en el plan; paso 0 con `completion-status` antes de
  bajar el servicio). RB-4: DEPLOYMENT ya no da por resuelto lo que piden
  SC4B-Q1 y SC4B-Q2.

Preguntas respondidas por el usuario el 2026-11-16 (recomendación aceptada en
las dos; `user-decisions-20261110.json`, clave `sc4b_questions_20261116`):

- **SC4B-Q1.** ¿Se puede hacer la primera transición posterior a R1 antes de
  tener el testigo de cierre por registro de recuperación y
  `--accept-regressed-source` (fila PA)? Recomendación: sí, si ocurre
  mientras L4 existe (14 días desde R1); si va a ser después, implementar
  antes esa parte.
- **SC4B-Q2.** ¿Se acepta para la primera transición que el retiro exija un
  escaneo de resurrección absoluto? Recomendación: sí, con el paso 0 del
  runbook (`completion-status`, `resurrected` en 0) antes de programar la
  ventana; si no da 0, implementar antes la línea base salada de K y el
  reencolado.

## SC-4c — lista cerrada de colecciones de Chroma en la compactación (2026-11-16)

Quinta unidad del modo release (diseño `t040-t041/sc-release-design-v4.json`,
unidad SC-4c: `mechanism.start.chroma_allowlist`, PRIV-3, PA-8, V4R2-C1).
Rama `sc4c` desde `batch-c` (e6d3239), sin push; la suite batch, la
revisión independiente (SC-R) y el merge los hace la sesión coordinadora.
Sin release. Evidencia: `t040-t041/sc4c-evidence.json` (`follow_up`,
pruebas, mutantes, pregunta abierta). Prioridad del usuario: sólo lo que
necesita la primera actualización posterior a R1.

- Hallazgo de partida: la copia idéntica de 0c rechazaba el snapshot de
  cualquier H con `data/chroma_db` (`state_snapshot_unregistered_sdk_root`):
  sólo `data/memory_db` era almacén SDK registrado. Producción tiene los dos.
- `src/memory/state_snapshot.py`: `snapshot_closed_state(...,
  chroma_collections=...)` trata todo almacén de Chroma de H como raíz SDK,
  lista sus colecciones en una copia aislada, clasifica cada nombre antes
  de exportar nada y no lee la colección dada de baja (sólo su nombre).
  Sin el argumento, el comportamiento anterior no cambia
  (`collection_records` compartido con `_export_indexes`).
- `src/memory/owner_compaction.py`: `collection_kind` (lista cerrada),
  `policy_allowlist` (lo que `get()` devolvería en t_clone, dueño no
  retirado, registro de borrados como segunda compuerta, revisión exacta),
  `chroma_plan` en el plan del clon (`plan_sha256` incluye resúmenes sin
  contenido), `_clone_state` (sin los bytes de Chroma de H),
  `rebuild_chroma` (cada almacén lo escribe un proceso hijo `-I -B` sin
  `src` que recibe sólo lo que K conserva y devuelve el sha256 de lo que
  lee de vuelta; el extent queda en el catálogo de copias; pin sha256 de
  los archivos) y D5 con esos pins antes y después del VACUUM y en
  `resume` (evidencia `chroma.pins`). El hijo existe porque en el server un
  `data_level0.bin` escrito desde el trabajador tuvo el texto de un hecho
  vencido (heap del proceso en la capacidad sin usar del índice HNSW).
- `scripts/compaction.py`: `chroma_collections` en `plan`/`start` (nombres
  de colección desde imágenes privadas de cada `chroma.sqlite3`, destinos
  `semantic_index` de H); `unlisted_collection` con almacén y colección en
  el informe; `chroma` en la vista del plan. La lista está duplicada a
  propósito (el controlador no importa `src`): una prueba fija que las dos
  coinciden.
- Tests: `tests/unit/tools/test_sc4c_chroma_closed_list.py` (paridad de la
  lista, canarios de la lista permitida, rechazos del plan del clon, la
  primera transición posterior a R1 con almacenes reales hasta el retiro,
  `user_memories` en el plan y en el trabajador, D5 de un almacén
  reconstruido, crash después de la evidencia y `resume`); estados en
  `tests/mocks/sc4c_chroma.py`. Mutantes: `tools/sc4c_mutants.py` (7).
  Resultados del server en la evidencia.
- Pregunta abierta SC4C-Q1 (diseño: «si SC-0 encuentra un lector, SC-R
  decide»): SC-0 encontró que «listar mis rutinas» lee
  `home_assistant_routines`. Esta unidad implementa la baja; ver la
  evidencia para la recomendación. SC4C-Q2: qué hacer si producción tiene
  la colección heredada `user_memories` (el plan rechaza). SC4C-Q3: el
  mismo residuo de heap en los HNSW que escribe `kza-voice` (anterior a
  esta unidad).

### SC-4c — revisión independiente, ronda 1 (2026-11-16)

Veredicto `changes_needed` (SC4C-RV-1 a RV-5); todo aplicado en la rama
`sc4c` (commits por hallazgo, sin push), evidencia en
`t040-t041/sc4c-evidence.json`, clave `review_round_1`.

- RV-1 (medio): el hijo de reconstrucción sobrevivía a un SIGKILL del
  trabajador y recreaba el almacén de K después del `abort`. Ahora muere
  con el trabajador en Linux (`PR_SET_PDEATHSIG`), sale antes de cada
  escritura si quedó huérfano, hereda el lease de admisión (todo rechaza
  `runtime_admission_busy` mientras vive) y tiene timeout de 600 s. Tests
  con un trabajador real matado con SIGKILL a mitad de la reconstrucción.
- RV-2 (medio): un test espía fija el hijo limpio (argv, entorno, lease,
  timeout, stdin sólo con lo planeado; el propio hijo informa que no cargó
  nada del repositorio). RV-3 (bajo): tests del hash de la imagen del hijo
  (inyección de fallas) y de los pins de la evidencia en `resume`.
- RV-5: nombres exactos en la lista cerrada (`<colección>_policy_v1`,
  `__pointer`, staging versionado); error del registro de borrados con
  código; `SQLITE_TMPDIR` para el hijo; `other_files` por almacén en el plan.
- Mutantes de la ronda en `tools/sc4c_mutants.py` (`REVIEW_1`).

### SC-4c — respuestas del usuario y baja de la `user_memories` heredada (2026-11-17)

Respuestas del 2026-11-16 (`t040-t041/user-decisions-20261110.json`, clave
`sc4c_questions_20261116`): SC4C-Q1 `home_assistant_routines` se da de baja;
SC4C-Q2 el listado de solo lectura de producción dio `data/chroma_db`
{`home_assistant_commands`: 337, `home_assistant_routines`: 0} y
`data/memory_db` {`user_memories`: 876, `user_memories_policy_v1`: 130};
SC4C-Q2b la heredada `user_memories` se descarta al actualizar; SC4C-Q3 el
residuo de heap en los HNSW de `kza-voice` va después, como unidad propia
con prioridad alta; SC4C-Q4 se mantiene el límite de 600 s y SC-6b lo
ajusta al medir; SC4C-Q5 (valor por defecto) el plan cuenta `other_files`
y el runbook pide revisar si no es 0.

- `src/memory/owner_compaction.py` y `scripts/compaction.py`:
  `CHROMA_DROPPED_IN_STORE = {("data/memory_db", "user_memories")}`, baja
  con nombre exacto y sólo en el almacén de memoria (el par de
  `memory.chroma_path`/`memory.long_term_collection`, fijado por un test).
  El snapshot no la lee, el hijo no la recibe; el plan nombra las bajas por
  almacén (`dropped_collections`) y el evento `materialized`/la evidencia
  de K las guardan como `[almacén, colección]`. Un nombre parecido, la
  misma colección en otro almacén o una `user_memories` registrada como
  destino siguen rechazando `unlisted_collection`.
- Tests nuevos en `tests/unit/tools/test_sc4c_chroma_closed_list.py`:
  transición completa con la colección heredada (canarios en documento y
  metadata; ningún byte en el estado de K ni en los registros de la
  compactación; ausente de toda la raíz después del retiro), rechazo del
  plan con un nombre parecido, otra mayúscula u otro almacén, paridad con
  `config/settings.yaml`. Mutantes `SC4C_Q2B` en `tools/sc4c_mutants.py`
  (el corredor ahora usa `taskset -c 24-47`, la casa está ocupada).
- Pendiente para el operador: el `compact plan` de solo lectura en la
  ventana de R1 debería mostrar `data/memory_db` con `dropped: 1`
  (`["user_memories"]`).
- Test inestable corregido (también fallaba en `49e3f8e`, antes de este
  cambio): las pruebas del lease de RV-1 daban por terminado al hijo cuando
  su hilo principal ya era zombi pero los otros hilos del SDK seguían
  saliendo con la tabla de descriptores, y con ella el lease; `abort` (o el
  cierre del propio `start`) veía `runtime_admission_busy`. El ayudante
  `_gone` ahora espera también a los hilos; mismas aserciones y plazos.

### SC-4c — revisión independiente, ronda 2 (2026-11-17)

Veredicto `changes_needed` (SC4C-RV2-1 a RV2-4) sobre `07b4a5b1`; sin
cambios de producto.

- RV2-1: el test de la `user_memories` heredada no revisaba el snapshot
  (sólo el registro de la compactación, `admission/compaction/v1/<id>`), y
  el test, `87d96f6a` y el informe decían que sí. El snapshot
  (`<raíz>/compaction/<id>/snapshot-n`) guarda una copia opaca de los
  archivos del SDK de H y con ella los bytes heredados hasta el retiro o el
  `abort`. Ahora el test toma la ruta del evento `snapshot` y afirma que los
  bytes están en su `state/data/memory_db/chroma.sqlite3`, sólo en archivos
  `sdk_opaque` del manifiesto e idénticos al archivo de H en la misma ruta
  (nunca en `manifest.json` ni en una exportación) y que fuera de H no hay
  otro lugar de la raíz con ellos; corre con dos finales, retiro y `abort`
  después de la activación (H idéntico y sirviendo, snapshot borrado,
  bytes sólo en H). La primera corrida en el server mostró una vez los
  bytes heredados también en un `data_level0.bin` de `data/chroma_db` de H
  y de su copia: residuo de heap del primer run del mundo de prueba
  (SC4C-Q3); el test lo admite sólo como copia idéntica de H.
- RV2-2: DEPLOYMENT y el runbook de compactación dicen ahora que el
  snapshot guarda esa copia opaca (`data/memory_db/chroma.sqlite3` y, en
  producción, los HNSW e `index_metadata.pickle`) y que `retire` y `abort`
  la borran.
- RV2-3: la prueba del lease con el hijo pausado vuelve a afirmar que el
  hijo vive (`process_started_at(pid) is not None`), no sólo que no se fue.
- RV2-4: sin acción (comprobaciones del revisor de baja exacta por almacén,
  crash, `resume` y `abort`).
- Server, como `yo`, `taskset -c 24-47 nice -n 10`, con `pgrep` < 4:
  `tests/unit/tools/test_sc4c_chroma_closed_list.py` 26 passed; el test
  heredado tres veces más (2 passed cada vez); los cuatro mutantes `SC4C_Q2B`
  que usan ese test, 4/4 muertos.

### SC-4c — SC4C-Q6: la baja de `home_assistant_routines` sólo en `data/chroma_db` (2026-11-17)

Respuesta del 2026-11-17 (`t040-t041/user-decisions-20261110.json`, clave
`followups_20261117`): «Sí, limitarlo (Recomendado): data/chroma_db only».

- `src/memory/owner_compaction.py` y `scripts/compaction.py`: desaparece
  `CHROMA_DROPPED` (la baja por nombre en cualquier almacén);
  `CHROMA_DROPPED_IN_STORE` tiene los dos pares,
  `("data/chroma_db", "home_assistant_routines")` y
  `("data/memory_db", "user_memories")`, y `collection_kind` sólo da
  `dropped` con almacén y nombre exactos. `home_assistant_routines` en otro
  almacén (por ejemplo `data/memory_db`) rechaza
  `runtime_admission_compaction_unlisted_collection` en el `plan` y en el
  trabajador (clasificador del snapshot y plan del clon). Producción no
  cambia: la colección está en `data/chroma_db`.
- Tests en `tests/unit/tools/test_sc4c_chroma_closed_list.py`: la tabla de
  la lista cerrada agrega nombres parecidos a las rutinas y almacenes
  parecidos a `data/chroma_db`; el par de rutinas se fija a
  `vectordb.path`/`vectordb.routines_collection`, al valor por defecto de
  `ChromaSync` y a la ruta por defecto de `src/main.py` (que no pasa
  `routines_collection`), y el conjunto de bajas a los dos pares de
  `config/settings.yaml`; rechazos del plan del clon con la colección en
  `data/memory_db`, omitida o leída; rechazo del `plan` (casos
  `routines_other_store`, `routines_look_alike`) y del trabajador (caso
  `routines_other_store`, fila S y `abort` sin pérdida).
- Mutantes `SC4C_Q6` en `tools/sc4c_mutants.py` (seis); los mutantes
  anteriores que nombraban `CHROMA_DROPPED` se reescribieron sobre el
  clasificador nuevo.
- Server, como `yo`, `taskset -c 24-47 nice -n 10`, con `pgrep` < 4, clon
  `/home/yo/projects/runs/sc4c-q6`, registros en
  `/home/yo/projects/runs/sc4c-q6-out/`, sobre `1bf0129d`:
  `test_sc4c_chroma_closed_list.py` 30 passed; `test_owner_compaction` y
  `test_state_snapshot` 34 passed; `test_compaction_units` 273 passed;
  los siete `test_stage_runtime_compaction*` 216 passed, 5 skipped (los
  cortes con víctima `snapshot` sin snapshot parcial, como antes);
  `test_sc1b_release_plan` 15 passed; `test_sc4a_verify_run` y los dos
  `test_sc4b_*` 108 passed. Mutantes (clon `sc4c-q6-mut`): los seis
  `SC4C_Q6`, los doce de la baja y del rechazo reescritos o afectados
  (`SC4C_Q2B` completos y los de rutinas y colección no listada de la
  lista original), 18/18 muertos (`mutants.json`).

## SC-5b — runbook del modo release (2026-11-17)

Rama `sc5b` desde `1541909a` (batch-e). Diseño `sc-release-design-v4`,
unidad SC-5b. Evidencia: `t040-t041/sc5b-evidence.json` (local).

- Runbook nuevo [2026-11-17-release-mode.md](2026-11-17-release-mode.md):
  compuerta (SC-R, SC-6b, SC4B-Q1 dentro de los 14 días de L4, SC4B-Q2 con
  el paso 0), construcción de NUEVO (toda release que cambia `src` lleva T0
  y revisión subida; `RELEASED_CONTROLLERS`; arreglo sobre la línea
  instalada), T0 con `"$PY" -I -B` desde el archivo extraído sólo con
  triple mayor, H servido una vez y unidad detenida y deshabilitada antes
  del plan, revisión del plan con la lista cerrada de Chroma, V, retiro
  ruteado por código, salidas (AT, SS con la confirmación escrita de SC-D8
  A, HD terminal sin L4, AB con el digest de la re-comprobación, CI),
  rollback sin T0, después de la aceptación (PA, `adopt`), DL/DO, hueco
  SC-D5 A y la plantilla de la decisión de 48 horas.
- DEPLOYMENT: sección «Procedimiento del modo release (SC-5b)», forma del T0
  corregida (`$PY -I -B`), estados de SC-1a/SC-4b. Runbook de compactación:
  el binding de H se lee del `intent.json` del log (la vista del plan nunca
  lo imprimió, como decía el paso 2).
- Código: fuera `_DELEGATION_OWNER_LIMIT` y la rama de concesión heredada
  del CLI `delegate` (rechaza siempre con 78), con sus tests.
- Al cerrar SC-5b fallaban cerrado PA (sin testigo por registro de
  recuperación ni `--accept-regressed-source`) y DR (la copia de
  `adopt --dry-run` en un temporal del sistema). Las dos filas están
  implementadas desde la unidad SC-PA-DR (DEPLOYMENT, «Filas PA y DR
  (SC-PA-DR)»; el runbook ya las describe): un head cerrado por un registro
  de recuperación limpio es fuente de una transición, uno con regresiones
  confirmadas sólo con `--accept-regressed-source`, y la copia del dry-run
  vive sólo bajo `<raíz>/admission/dry-run/` y la borra el próximo que toma
  la concesión. El dry-run corre siempre que no haya un log del modo
  release pendiente: también en T0F y en el `not_closed` del T0, durante la
  primera transición.
- **Plazo de PA y DR (SC5B-Q1 y SC5B-Q2, respuesta del usuario del
  2026-11-17):** las filas PA y DR se implementan antes de DÍA 0 + 14 × 24 h
  de R1; ya están en el código (SC-PA-DR), y para una transición después
  del día 14 tienen que estar en el controlador instalado, revisadas (SC-R)
  y ensayadas (SC-6b); sin las dos, ninguna transición después del día 14.
  Está en la compuerta del runbook y en DEPLOYMENT («Filas PA y DR,
  implementadas»).
- Revisión independiente, ronda 1 (2026-11-17): `changes_needed`, 3
  hallazgos medios (SC5B-R1 a R3), 6 bajos (R4 a R9) y 2 informativos (R10,
  R11 sin cambio), cada uno en su commit. CI se instala sin servir H (arrancar
  la unidad recuperaría K'); `io_error` del retiro se rutea por fase (después
  de `retire_started` sólo `retire` o `resume`); `adopt --dry-run` también
  corre en T0F y en el `not_closed` del T0, con la limpieza ahí; fila por
  defecto del retiro; `retire` en vez de `resume` para enlaces; `verified`
  diferido va primero al retiro; el desajuste de descarte con `equal` o
  `loss_only` saca `--discard-k-changes`; L4 trae de vuelta los datos
  borrados (PA y DO); «Compactación trabada» con la forma extraída. Los 16
  mutantes de la revisión sobrevivían al test de docs; ahora 32 de 32
  muertos (los 16, uno más de CI, 14 de los arreglos y un control). Detalle
  en la evidencia (`review_round_1`).
- Revisión independiente, ronda 2 (2026-11-17): `changes_needed`, 2
  hallazgos medios (SC5B-RR1, RR2), 3 bajos (RR3 a RR5) y 1 informativo
  (RR6), cada uno en su commit. Con V muerta después de `READY` y K'
  abierto, el retiro rechaza `runtime_admission_not_closed` sin escribir:
  fila V2, volver a H con W (el retiro primero vale sólo con K' cerrado);
  frases de seguridad fijadas enteras; DEPLOYMENT y handoff con dónde corre
  el dry-run y el plazo del día 14; un rechazo del retiro después de
  `retire_started` es un retiro cortado; el runbook de compactación repite
  `compact retire` ante un enlace; un enlace duro nunca quita un nombre bajo
  `slots/<K>/`. Mutantes: 34 de 34 muertos (los 15 de la revisión, 5 de la
  compuerta, un control y 13 de los arreglos). Detalle en la evidencia
  (`review_round_2`).

## SC4C-Q3 — sin residuo de heap en los índices de Chroma del servicio (2026-11-18)

Prioridad alta por decisión del usuario (SC4C-Q3, 2026-11-16). Rama
`sc4c-q3` desde `batch-f` (`1008cc1e`). Mecanismo medido en el server: Chroma
1.5.5 escribe enteros los buffers `malloc` de un índice HNSW al
inicializarlo (`data_level0.bin`, `length.bin`) y lo vuelve a inicializar en
cada proceso que abre un segmento sin mapa de ids persistido (menos de 1000
operaciones: el caso de producción), así que cada apertura de `kza-voice`
reescribía esos archivos con su heap; un canario sembrado llenó
`data_level0.bin` tras un `count()`. Corrección: el launcher le da al hijo
administrado `GLIBC_TUNABLES=glibc.malloc.perturb=165:glibc.malloc.tcache_count=0`
(lo mismo el hijo limpio de la compactación) y `src.main`, con ticket y en
glibc, sale con 78 antes de armarse si su `malloc` no devuelve el patrón
(`runtime_admission_heap_hygiene_inactive`). Detalle, límites y costo en
DEPLOYMENT («Residuo de heap en los índices de Chroma (SC4C-Q3)»);
evidencia `data/autonomous-roadmap-20261102/sc4c-q3/evidence.json` (local,
fuera de Git).

Revisión independiente, ronda 1 (2026-11-18): `changes_needed`, cuatro
hallazgos, aplicados uno por commit. R1 (alto): `AdmissionController.run`
(`run_slot`) arrancaba un `src.main` con ticket sin los tunables y el ensayo
compilado `test_compiled_stop_contract.py` salía con 78; ahora les pone los
mismos que el launcher (test nuevo). R2: el costo en memoria (RSS) y en
arranque, medido y anotado en DEPLOYMENT; la carga real de modelos se mide
en la ventana de SC-6b. R3: un controlador anterior a Q3 con un slot de Q3
lo cierra el triple del controlador (límite 7 en DEPLOYMENT y nota en el
runbook del modo release). R4: redacción («nunca contenido anterior»).
Detalle en la evidencia (`review_round_1`).

## Prompt para la sesión nueva

Usar el prompt completo de [cierre2026-11-07](2026-11-07-session-handoff.md#prompt-para-sesión-fresca-y-goal-nuevo). El goal anterior está pausado; crear uno nuevo en el hilo fresco para el roadmap completo. No repetir el hito S-only ni iniciar cuentas antes de reparar la full actual.

## Lista de cierre y reanudación

- [x] Linux y unidades G/H/I/J conservados con recibos propios.
- [x] T024-A/B/C/D conservadas y T024-E con original1, dirigida1230, full y limitaciones documentados.
- [x] T024-F0: RED original, green4, matriz6, dirigida/full y revisión independiente conservados.
- [x] F1 original: RED genuino, tres cierres/waits0, archivo privado y revisión independiente conservados; sin fix de producto.
- [x] Código, tracking y handoff guardados; verificar publicación y HEAD al retomar.
- [x] Goal anterior pausado por pedido del usuario, no completado; sesión nueva con goal nuevo y criterios físicos pendientes.
- [ ] Completar F1: censo y owners globales, cierre SDK genuino, migración legacy y guards pendientes; los avances locales anteriores no cierran esa aceptación.
