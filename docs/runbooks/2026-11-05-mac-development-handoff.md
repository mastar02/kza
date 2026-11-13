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

Siguiente:
- U3b-0e2: biblioteca de adopción (copia, migración de solo esquema con VACUUM, reubicador, sembrado del registro de borrados y restos con reapertura del paso).
- U3b-0e3: comandos.
- U3b-0c: compactación.
- F8.

Decisiones del usuario en `t040-t041/user-decisions-20261110.json` (`u3b_*`).

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
