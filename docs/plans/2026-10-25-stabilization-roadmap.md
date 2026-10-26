# Roadmap de estabilización y revisión de voces de KZA

El resultado buscado es una única release que controle los dispositivos elegidos
con una política común, responda por voz sin bloquear la captura, conserve la
memoria durante los plazos acordados y permita revisar las voces agrupadas por
probable hablante. El desarrollo avanza por hitos verificables; terminar un hito
no implica publicarlo en producción.

Este roadmap ordena los [pendientes contrastados](2026-10-25-review-followups.md)
y agrega la vista de voces solicitada el 2026-10-25. Los detalles de cada cambio
se concretan en su plan de implementación al empezar el hito. Las fechas se
estiman después de cerrar el inventario y los contratos de H0; la secuencia no
supone una duración medida que todavía no tenemos.

## Decisiones acordadas

| Tema | Decisión |
|---|---|
| Publicación | Una sola release al completar todos los frentes incluidos. |
| Voz sin identidad verificada | Control de luces, escenas, interruptores, clima y persianas elegidos. |
| Acciones sensibles | Cerraduras y alarmas bloqueadas; la similitud de voz no eleva permisos. |
| Dashboard | Acceso desde la red de casa mediante el proxy del servidor, siempre autenticado. |
| Ambient | Se conserva, con pausa y borrado efectivos. |
| Texto ambiental | Retención de 48 horas. |
| Hechos extraídos automáticamente | Retención de 30 días. |
| Recuerdos confirmados | Se conservan hasta borrarlos. La voz propone y el dashboard autenticado confirma. |
| Hechos legacy sin procedencia verificable | En cuarentena, fuera de las respuestas automáticas, hasta revisión manual; decisión del usuario del 26 de octubre. |
| Borrados amplios de memoria | Se autorizan desde el dashboard autenticado. |
| Vista de voces | Grupos sugeridos, asignación de nombre y corrección de uniones o separaciones. |
| Audio para revisar voces | Fragmentos locales que caducan a las 48 horas. |
| Música | Fuera de este roadmap. |
| Avisos externos de sordera o caída | Postergados. Se mantienen recuperación automática y diagnóstico local. |

Las escenas y rutinas deben respetar las mismas restricciones que una orden
directa. Elegir `climate` o `cover` no habilita todo el dominio ni sus servicios:
se revisan entidades, operaciones y parámetros concretos. Los nombres de
personas, sus audios, embeddings y datos del hogar permanecen fuera de Git.

## Hitos y dependencias

| Hito | Resultado | Pendientes cubiertos | Dependencias |
|---|---|---|---|
| H0 | Baseline del rescate y contratos de implementación | R09 y preparación de R10 | Ninguna |
| H1 | Control autorizado y semántica segura | R01, R02, R03 y consentimiento de R08 | H0 |
| H2 | Respuestas completas y captura disponible | R04 y R05 | H1 |
| H3 | Privacidad y revisión de voces | R08 y R11 | H1; integra la reproducción corregida en H2 |
| H4 | Recuperación y diagnóstico local | R06 | H0; integración con H2 |
| H5 | Rutinas, timers, recordatorios y alertas locales | R07 dentro del alcance acordado | H1 y H2; integra H3 y H4 |
| H6 | Candidato integrado y única release | R09, R10 y validación de todos los hitos | H1 a H5 completos |

El orden de ejecución recomendado es H0, H1, H2, H3, H4, H5 y H6. Las pruebas,
la instrumentación y la documentación de cada comportamiento acompañan su
implementación; no se acumulan para H6.

### H0 Fijar baseline y contratos

En ejecución: [baseline, contratos y ensayo de release](2026-10-25-h0-baseline-contracts.md).
Los diez fallos están clasificados y el staging tiene pruebas de restauración
conjunta de runtime y datos. Se recibió el inventario privado persistido de HA
y se registró la selección inicial del usuario fuera de Git. Quedan pendientes
la evidencia viva y los efectos sin resolver del inventario. Los metadatos
legacy tienen inventario de sólo lectura y vista previa privada; el usuario eligió
conservarlos en cuarentena. La prueba de migración en copia sigue pendiente.
H0 todavía no está cerrado.

**Entregables:** clasificación de los diez tests fallidos del rescate, matriz de
flujos realmente conectados y plan probado de instalación, promoción y rollback
en un directorio de staging bajo `yo`. Reutilizar el builder CPython 3.13.

Preparar un inventario privado de las entidades disponibles y sus capacidades,
incluidas escenas y automatizaciones invocables. Revisar cómo se representa una
orden, quién la produce y cómo se registra su resultado. Separar en ese contrato
actor autenticado, voz no identificada, perfil de hablante y agrupación acústica.

**Salida:** cada fallo tiene causa y prueba responsable identificadas; se conoce
la diferencia entre contratos nuevos y tests obsoletos. Los planes de H1 y H3
pueden definir interfaces y migraciones sin inventar identidades ni asumir que
una respuesta HTTP confirma un efecto físico. El baseline conserva los fallos
abiertos explícitos hasta corregirlos.

### H1 Cerrar las vías de control y las órdenes falsas

Plan abierto: [control autorizado y semántica segura](2026-10-25-h1-authorized-control.md).
La implementación semántica tiene un corpus compartido y rechazos conectados
en ambos routers; la captura espera el endpoint y comprueba la cobertura del
texto reutilizado. HTTP/WS privados ya comparten autenticación y la UI verifica
la sesión antes de cargar datos. Los consumidores de inferencia aplican destino
de procesamiento y consentimiento explícitos, también detrás de gateways
loopback. Los productores de controles comparten servicio, actores de procedencia
y selección privada revisada antes del envío, incluidas acciones diferidas.
Los contratos, servicio, formularios y ambos routers de voz cubren operaciones
ampliadas con interpretación determinística y selección exacta por habitación.
La expansión de grupos y escenas ya tiene pruebas conectadas con hojas
autorizadas y recibos parciales. Siguen pendientes la membresía física y la
verificación de efectos indirectos del inventario; configuración y validación
acústica reales se contrastan durante H6.

**Entregables:** protección uniforme de HTTP y WebSocket, bind local detrás del
proxy y una política de acciones compartida por voz, dashboard, rutinas y
recordatorios. La política queda activa aunque se deshabiliten hooks opcionales.

La lista permitida incluye los equipos seleccionados de clima y persianas.
Extender las operaciones tipadas y su confirmación antes de exponerlas: el
`HomeActionService`, formularios y voz ya tienen contratos ampliados; falta
verificar las definiciones y los efectos reales antes de habilitarlas. Validar parámetros
contra las capacidades de cada entidad; no convertir la salida libre de un LLM
en un servicio arbitrario de HA. Una escena no puede saltar la política por
ocultar una acción restringida; si sus efectos no se pueden verificar, se
rechaza su ejecución desde KZA.

Completar el corpus de negaciones, pasado, preguntas y acciones diferidas. Hacer
explícito el destino cloud aun cuando el gateway tenga URL loopback.

**Salida:** tests conectados demuestran que no hay llamadas HA sin autorización,
incluidos envíos diferidos y escenas. Cerraduras y alarmas quedan bloqueadas.
`¿Apagaste la luz?` no ejecuta una orden y los pedidos inmediatos válidos siguen
funcionando. HTTP y WebSocket privados rechazan sesiones ausentes u orígenes
inválidos; el consentimiento denegado impide el envío al cloud en sus consumidores.

### H2 Completar generación y reproducción de voz

Plan abierto: [respuestas completas y captura disponible](2026-10-25-h2-complete-voice-responses.md).
Los límites offline y la validación acústica pendiente se fijan allí antes de medir.
La capa PCM conserva los samples y espera drenado en grafos sintéticos. El
worker encolado ya consume HTTP asíncrono y un adaptador acotado para motores
síncronos; comparte la cola del dispatcher, cancela lecturas pendientes y
descarta resultados tardíos del historial. Ambos routers y el worker usan un
propietario de respuesta, cola de frases acotada y PCM con cancelación para
cache, síntesis completa, streaming, earcons y broadcast. El factory usado por
`main` conecta también productores de features y síntesis de arranque/cache al
propietario, con cierre de admisión y espera de recursos reales. El callback de
captura tiene un caso conectado de barge-in sobre salida simulada. La revisión
del 26 de octubre incorpora presupuestos por etapa, timeout durante entrega
parcial y autoridad única para el error, incluso al cancelar durante su cleanup
o drenado. El corpus incluye una respuesta larga completa y cierres nativos
retenidos sin abrir conexiones sucesivas. Sigue pendiente aceptar el grafo
completo con runtime y hardware, incluidas las pausas globales de GC observadas
con el heap de la suite. La medición aislada del grafo no cierra H2.
La [validación del 26 de octubre](../research/2026-10-26-h2-response-deadline-validation.json)
registra 4199 pruebas pasadas y el fallo conocido de VAD/idioma pendiente de H3.

La [revisión del transporte nativo](../research/2026-10-26-h2-native-http-validation.json)
del commit `83d064f` elimina la inicialización perezosa de tipos del SDK de
`astream()` y `complete()`, que provocaba la pausa reproducida durante GC.
Conserva consentimiento, modelo, validación de respuesta y cierre real; ambas
entradas usan HTTPX, con parsing acotado. La suite completa pasa **4405 tests,
1 xfailed**, sin fallos, en 156,38 s; el heartbeat alcanza p95 de 0,25 ms y máximo
de 1,07 ms frente a 25/75 ms. Pasan los checks y el artefacto compilado. Esto
corrige ese disparador concreto; las pausas globales bajo otras cargas y la
aceptación física siguen pendientes de H4/H6. No cierra H2.

**Entregables:** consumo no bloqueante del reasoner, un único responsable de
locutar cada respuesta y reproducción que conserve todos los samples. Integrar
cancelación, presión de retorno al productor y mensajes de error audibles.

**Salida:** una consulta conversacional larga no congela la captura, no se
locuta dos veces y puede cancelarse. Chunks mayores que el buffer no pierden
audio. Timeout, caída del gateway y fallo de reproducción producen estados
coherentes. Validar una respuesta no cacheada y barge-in con el micrófono y la
salida reales durante la validación de release. Medir la latencia del loop y la
cancelación; fijar sus límites en el plan antes de evaluar el resultado.

### H3 Aplicar privacidad y revisar voces

Plan concreto del avance de memoria:
[procedencia, retención y revisión](2026-10-26-h3-memory-privacy.md).

Decisión del usuario del 26 de octubre: huellas y grupos sin confirmar vencen
a las 48 h; los perfiles confirmados manualmente se conservan hasta borrado
manual. Los clips y las transcripciones siguen venciendo a las 48 h. Nombrar
un grupo no le concede permisos; una sugerencia automática no equivale a una
confirmación manual. La implementación y la migración siguen pendientes.

Primer avance de H3, 26 de octubre: el store de capturas aplica TTL al leer
texto, lotes de destilación, estadísticas y clips, sin esperar la purga. Las
capturas nuevas tienen UUID y vencimiento persistido; ampliar la configuración
no renueva ese plazo. Las filas previas conservan sus metadatos, sin asignarles
UUID ni confirmaciones inventadas. El borrado de clips tiene una cola durable
de reintento, y el worker se programa para el próximo vencimiento, también al
arrancar. La escritura de archivo cancelada conserva ownership hasta su salida
real y descarta el resultado tardío. Son pruebas con SQLite/FLAC sintéticos;
no se migraron ni borraron datos privados reales.

El siguiente avance de H3 agrega un catálogo SQLite como autoridad de memoria,
con procedencia de las capturas nuevas, hechos automáticos de hasta 30 días y
propuestas que vencen con sus fuentes (hasta 48 h). La consulta no renueva esos
plazos. La voz propone; la pestaña Memoria del dashboard autenticado confirma,
descarta o elimina. Sólo la confirmación manual crea recuerdos sin vencimiento.
El índice Chroma nuevo excluye la colección legacy; el JSON de preferencias
legacy también queda fuera de los prompts y ambos originales se conservan.
Se implementan tombstones de conversación para impedir nuevas derivaciones en
el catálogo, y una cola durable de reconciliación con Chroma. Esto todavía no
constituye un borrado coordinado de conversación entre todos los almacenes.

El avance de contextos extiende la procedencia y el vencimiento a los turnos del
orquestador, los resúmenes y sus snapshots privados. El plazo de un resumen
depende de sus capturas, aunque se compacte nuevamente o se rehidrate otra
sesión. Fuentes vencidas o revocadas se excluyen de texto, chat, entidades y
serialización. Se rechazan resultados de compacción tardíos y escrituras que
llegan después del tombstone; un snapshot no elimina un contexto que volvió
a activarse mientras esperaba. Los archivos nuevos viven separados de los JSON
legacy, que no se importan ni se modifican. La carga, escritura y purga de esos
snapshots tienen seguimiento de trabajo nativo fuera del loop.

El registro SQL de eventos ahora conserva fuentes y deadlines por evento.
Transcripciones, nombres copiados y contexto libre vencen con la captura,
hasta 48 h; los metadatos de actividad y señales numéricas conservan su plazo
configurado de analítica. Las consultas de historial, estadísticas y secuencias
filtran vencimientos y revocaciones, y el worker retira el texto vencido.
La tabla legacy permanece fuera de esas consultas y conserva sus filas.
Ambos routers adjuntan la captura original incluso sin catálogo de memoria.
El diagnóstico tiene espera acotada y seguimiento del trabajo nativo; la API
autenticada consulta fuera del loop y responde con `Cache-Control: no-store`.

Las solicitudes, la caché de consultas y el historial del clasificador ahora
conservan sus capturas y plazos originales. Se retiran cuerpos y closures
vencidos o revocados; una generación activa se cancela si pierde su fuente.
Las entradas históricas incluidas en el prompt del clasificador también se
comprueban al iniciar y terminar el trabajo nativo. Ambos routers registran
una vez los controles aceptados. El mantenimiento en inactividad y el cierre
pertenecen al pipeline; búsquedas y clasificaciones tardías no repueblan la
caché cerrada ni publican el contenido invalidado.

El seguimiento de prompts activos conserva ahora las capturas incluidas y los
recibos de hechos recordados. Ambos modos de voz transmiten sus dependencias al
intercambio registrado. Borrar un padre semántico invalida sus derivados; leerlo
no renueva el plazo. Se comprueba admisión después de discovery, por fragmento
y durante la entrega. Limpiar o reemplazar un contexto también invalida el prompt
en curso. El catálogo publica revocaciones antes de devolver el commit, conserva
cleanup ante cancelaciones repetidas y no confunde un commit incierto con un
rollback probado. Los snapshots nuevos usan versión 3; los resúmenes de versión
2 quedan fuera de inferencia hasta su vencimiento original. La migración de
esquema, bajo lease, conserva en cuarentena los automáticos anteriores sin
dependencias verificables y mantiene las confirmaciones manuales.

La [validación de procedencia de prompts](../research/2026-10-26-h3-prompt-provenance-validation.json)
del commit `951bc4a` registra **4434 passed, 1 xfailed**, sin fallos, en 158,11 s.
Incluye 29 casos nuevos y pasa Ruff, Bandit medio/alto, 18 tests Node, smoke y
artefacto compilado. El heartbeat mantiene los límites de 25/75 ms: p95 de
0,19 ms y máximo de 1,02 ms. Estas pruebas usan datos sintéticos; los 870 hechos
legacy y producción permanecen sin cambios.

La autoridad de privacidad ya se inicializa aunque la memoria semántica esté
apagada. NLU, eventos, contexto, cola y captura comparten el catálogo; su pool
de I/O y su lease sobreviven al trabajo nativo real de la memoria opcional.
El transcriber fija fuentes antes del STT; el store ambiental excluye capturas
y clips revocados y conserva el reintento durable del unlink. Los scopes
rechazan capturas anteriores al borrado y buffers sin conversación anteriores
al arranque. La autoridad indisponible produce fallo de lectura/purga, no una
confirmación de borrado ni la retirada de snapshots desconocidos.

La [validación de autoridad independiente](../research/2026-10-26-h3-privacy-authority-validation.json)
del commit `74edec5` registra **4459 passed, 1 xfailed**, sin fallos, en 159,57 s.
Incluye 25 casos nuevos, conserva conversación y control local con memoria
apagada, y pasa checks y artefacto compilado. El heartbeat mantiene p95 de
0,23 ms y máximo de 1,09 ms frente a 25/75 ms. Ese paso todavía no incluía el
coordinador con journal y participantes, la limpieza de índices con memoria
apagada ni los lotes de destilación en curso; no expone un borrado global.

El avance siguiente agrega el journal transaccional de borrado y un manifiesto
obligatorio de diez participantes. Sólo está conectado el índice: la limpieza
de conversaciones registradas funciona sin recall, conserva destinos históricos
y cursores de lotes, y espera escrituras anteriores a la revocación. El catálogo
registra la ubicación antes del primer acceso semántico. Las operaciones siguen
bloqueadas por los participantes pendientes; cancelar la observación o agotar
el plazo de cierre no libera su trabajo nativo ni la lease. Faltan los demás
productores/copias y la integración autenticada del borrado global. La
reconciliación de todos los vencimientos y borrados individuales sin recall
también sigue pendiente. El detalle está en el plan de H3.

La [validación del journal](../research/2026-10-26-h3-erasure-journal-validation.json)
del commit `572aad5` registra **4484 passed, 1 xfailed**, sin fallos, en 161,58 s,
con 25 casos nuevos y los checks y artefacto compilado aceptados. El heartbeat
mantiene p95 de 0,18 ms y máximo de 1,00 ms frente a 25/75 ms. La revisión cubre
la retención sanitizada de resultados mientras falla el guardado de recibos.

La retención de huellas autorizada arriba todavía no está implementada. Siguen
pendientes otras copias de texto y audio (otros logs, colectores,
caches alternas, streams y frontend), sugerencias y consumidores alternos de
prompts —incluido `RequestProcessor`, antiguo y no cableado al runtime—,
el borrado completo de conversación,
pausa/mute, agrupación de voces, su vista autenticada y evaluación acústica, y la
migración ensayada sobre copia de datos reales. Este avance no cierra H3.

El participante de capturas ya espera encoders reales y elimina filas, scopes,
clips y localizadores durables de la conversación revocada. Una reserva anterior
al encoder permite encontrar archivos sin adjuntar tras un crash. Los clips
nuevos usan UUID para evitar colisiones entre bases, y las raíces históricas
registradas siguen incluidas. Los fallos de unlink, bases corruptas, directorios
ilegibles y clips sin una base atribuible conservan el estado pendiente. Quedan
ocho participantes sin conectar, el drenado de otros productores y lectores,
la purga de todos los destinos aun con ambient apagado y el borrado global en
el dashboard autenticado. No se modificaron datos reales ni producción.

La [validación de borrado de capturas](../research/2026-10-26-h3-capture-erasure-validation.json)
del commit `63495dc` registra **4507 passed, 1 xfailed**, sin fallos, en 162,85 s,
con 23 casos nuevos y checks y artefacto compilado aceptados. El encoder aplica
también el plazo menor del store cuando vence antes que la fuente. El heartbeat
obtuvo p95 de 0,39 ms y máximo de 1,24 ms frente a 25/75 ms. Las correcciones del
hook de cancelación y del vencimiento se incluyen en el candidato final.

El avance siguiente independiza la purga del ambient path: recorre bases y
raíces históricas aun con ambient, keep_audio y recall apagados, mantiene fechas
persistidas y despierta ante vencimientos anteriores. Usa dos barridos y dos
threads de disco propios; un destino bloqueado no frena otro independiente.
Los archivos compartidos se conservan mientras otra base los referencia o no
puede verificarse su estado. Las reservas, descartes y operaciones reales de
purga permanecen bajo la lease durante cierre. Quedan abiertos el inventario de
archivos legacy indeterminados, los ocho participantes restantes y el resto de
H3; no se expone un borrado global ni se modifica producción.

La [validación de purga independiente](../research/2026-10-26-h3-capture-maintenance-validation.json)
del commit `22ebc58` registra **4527 passed, 1 xfailed**, sin fallos, en 163,30 s,
con 20 casos nuevos y los checks y artefacto compilado aceptados. El heartbeat
obtuvo p95 de 0,40 ms y máximo de 1,05 ms frente a 25/75 ms. El ensayo compilado
comprueba vencimiento físico sin un tombstone de conversación ni ambient activo.

El mantenimiento del índice de hechos ya procesa también vencimientos y borrados
individuales con recall apagado. Los recibos se guardan por destino y revisión
en la transacción del catálogo; registrar otra ubicación incorpora pendientes
anteriores. El arranque con lease revalida los índices físicos y las operaciones
semánticas previas, sin confiar en que no se restauró una copia vieja. Comparte
la barrera del writer real y los recibos del borrado de conversación; versiones
o autoridades desconocidas quedan pendientes. La reconstrucción de índices
vivos, los otros ocho participantes y la restauración compatible siguen abiertos;
no se expone todavía un borrado global ni se modificó producción.

La [validación del mantenimiento del índice](../research/2026-10-26-h3-index-maintenance-validation.json)
del commit `5faa945` registra **4557 passed, 1 xfailed**, sin fallos, en 175,95 s,
con 28 casos nuevos del índice, dos del arranque del logger, checks y artefacto
compilado aceptados. El primer fallo de admisión en frío queda documentado con
su corrección y presupuesto de 100 ms intacto. Los casos de voz pasan, pero la
observación de GC del proceso de tests registra pausas de hasta 495,68 ms fuera
de ellos. La aceptación de runtime/GC de H4/H6 sigue pendiente; no se interpreta
esta suite como prueba de latencia integrada ni se desactiva GC para aprobarla.

El participante de eventos se conecta al journal aun con analytics apagado.
Recorre bases actuales e históricas en lotes de 128 metadatos y conserva las
filas ajenas y la cuarentena legacy. Las escrituras comparten la barrera de
revocación, y su trabajo nativo mantiene la lease incluso al vencer la espera
del cierre. Cada writer y adaptador usa dos threads propios; los providers ya
no comparten el pool de la autoridad. Siguen pendientes siete participantes y
las copias derivadas. La retención independiente se describe en el avance
siguiente; no se expone una confirmación global de borrado.

La [validación del participante de eventos](../research/2026-10-26-h3-event-erasure-validation.json)
del commit `4c297b3` registra **4582 passed, 1 xfailed**, sin fallos, en 169,95 s.
Pasan 25 casos nuevos, 132 pruebas dirigidas, los checks y el ensayo compilado
con borrado físico de SQLite y recibo de eventos completo. El heartbeat alcanza
p95 de 0,22 ms y máximo de 1,07 ms, con los límites intactos. La validación de
GC/latencia integrada continúa abierta; no se infiere de esta suite sin fallos.

La retención de eventos también se construye ya con analytics apagado y recorre
los destinos históricos en lotes de 128 metadatos. Respeta y persiste fechas
anteriores, despierta ante nuevas escrituras y conserva trabajo real durante el
cierre. El factory usa ese mantenimiento acotado en lugar de otro barrido del
logger. La limpieza física usa sólo revocaciones confirmadas: un commit fallido
puede bloquear lecturas provisionalmente, pero no destruir eventos válidos.
Siguen abiertos los siete participantes restantes, las otras copias, voces,
migración y aceptación de runtime; no hay cambios en producción.

La [validación de retención de eventos](../research/2026-10-26-h3-event-retention-validation.json)
del commit `d6dd9bc` registra **4607 passed, 1 xfailed**, sin fallos, en 176,39 s,
con 25 casos nuevos, 157 pruebas dirigidas, checks y artefacto compilado
aceptados. La revisión corrigió además dos huecos reproducidos en la lectura de
fuentes y fechas, con exclusión anterior a la purga física. El heartbeat obtiene
p95 de 0,21 ms y máximo de 1,73 ms, sin ampliar límites. La aceptación integrada
de runtime/GC y hardware continúa pendiente.

El participante de contextos ya conecta snapshots actuales/históricos de los
formatos de política y los buffers de contexto, memoria corta y consultas.
Conserva archivos ajenos/legacy y deja pendiente lo que no puede atribuir.
Las escrituras comparten barreras físicas y las compactaciones, hidrataciones y
guardados afectados se retienen hasta su salida real. La lease no se libera por
cancelar al observador. Quedan seis participantes y el resto de H3; el avance
siguiente cubre el TTL independiente de los features. No hay una promesa global
de borrado ni cambios en producción.

La [validación del participante de contextos](../research/2026-10-26-h3-context-erasure-validation.json)
del commit `5aec4fb` registra **4630 passed, 1 xfailed**, sin fallos, en 178,16 s,
con 23 casos nuevos, 462 pruebas dirigidas, checks y artefacto compilado
aceptados. El ensayo verifica snapshot y buffers eliminados con los features
apagados. El heartbeat obtiene p95 de 0,20 ms y máximo de 1,03 ms con límites
intactos. No cierra H3 ni la aceptación integrada de runtime/GC y hardware.

El vencimiento de contextos ya se mantiene con los features apagados, tanto en
snapshots históricos como en buffers registrados que nadie vuelve a consultar.
Los archivos atribuibles vencidos se retiran en lotes de 32; los desconocidos
permanecen pendientes sin frenar los demás. La actualización del registro no
bloquea otros directorios conocidos cuando un writer retiene el catálogo.
Las compactaciones pendientes tampoco detienen otros buffers. Siguen abiertos
los seis participantes restantes, los datos sin procedencia, la migración,
voces y la aceptación integrada; no se publica una release intermedia.

La [validación de retención de contextos](../research/2026-10-26-h3-context-retention-validation.json)
del commit `7606e42` registra **4652 passed, 1 xfailed**, sin fallos, en 183,55 s,
con 22 casos nuevos, 508 pruebas dirigidas, checks y artefacto compilado
aceptados. Verifica vencimiento sin consultas, tombstone ni gestor activo,
incluido un directorio limpio mientras otro writer y el registro siguen
bloqueados. El heartbeat obtiene p95 de 0,22 ms y máximo de 3,07 ms con los
límites intactos; no sustituye la aceptación integrada de runtime/GC y hardware.

La [validación de transitorios](../research/2026-10-26-h3-transient-privacy-validation.json)
del commit `85f5d39` registra **4324 passed, 1 failed, 1 xfailed** en 154,91 s.
Pasan los 18 casos nuevos, Ruff, Bandit medio/alto, 18 tests de frontend, smoke
autenticado y artefacto compilado. La suite completa **no está aceptada**: el
heartbeat llegó a 226,45 ms frente al máximo de 75 ms. La instrumentación
reprodujo una pausa global de GC de unos 227 ms durante las importaciones
perezosas de tipos del SDK en su worker. Ese camino se corrige y vuelve a medir
en la [revisión HTTP nativa](../research/2026-10-26-h2-native-http-validation.json)
indicada en H2, con suite posterior sin fallos. El registro original conserva
su fallo y los límites; el avance de privacidad no sustituye la aceptación de
latencia del runtime completo.

La [validación de eventos](../research/2026-10-26-h3-event-privacy-validation.json)
del commit `4ef72bf` registra **4307 passed, 1 xfailed**, sin fallos, en
156,43 s. Incluye 14 casos nuevos de eventos y dos de cierre de generación.
La primera ejecución completa detectó una carrera: se notificaba éxito antes
de liberar la petición actual. La corrección ordena la notificación después
del cierre de generación y de esa liberación; conserva la cancelación inmediata
con cleanup bajo seguimiento. Pasan Ruff, Bandit medio/alto, 18 tests de
frontend, el smoke autenticado y el artefacto compilado con eventos y TTL.

La [validación de contextos](../research/2026-10-26-h3-context-privacy-validation.json)
del commit `2bc494f` registra **4291 passed, 1 xfailed**, sin fallos, en
151,86 s. Incluye 16 casos nuevos de vencimiento, resúmenes tardíos, escrituras
ordenadas respecto del tombstone, cargas canceladas, reactivación de contexto
y presión de los ocho workers. Pasan Ruff, Bandit medio/alto y el artefacto
compilado con guardado, vencimiento y purga de snapshots. Los 18 tests del
frontend y el smoke autenticado también pasan; no hubo cambios de frontend.

La [validación de memoria](../research/2026-10-26-h3-memory-retention-validation.json)
del commit `cfa949c` registra **4275 passed, 1 xfailed**, sin fallos, en
151,10 s; incluye 25 casos conectados nuevos con SQLite y Chroma temporales,
ambos routers y HTTP/SSE sintético. Pasan Ruff, Bandit medio/alto, 18 tests de
frontend, el smoke autenticado, Chromium en escritorio y móvil y el artefacto
compilado con el servicio de memoria. No es evidencia de migración real ni de
evaluación acústica o release.

La [validación de capturas](../research/2026-10-26-h3-capture-retention-validation.json)
del commit `e186430` registra **4226 passed, 1 xfailed**, sin fallos, en
145,41 s. Pasan los 26 casos nuevos, Ruff, Bandit medio/alto, los doce tests
del frontend, el smoke autenticado y el artefacto compilado. El caso histórico
de VAD queda resuelto con conservación de texto y filtrado de destilación
probados por separado; el xfail documentado del extractor regex se conserva.

El análisis de patrones de actividad ya conserva recibos verificables de sus
eventos y usa una ventana común de metadatos sin cargar texto privado. Corrige
el conteo repetido de disparos en secuencias y se abstiene ante historial
incompleto; revalida fuentes y plazos después del cálculo. La generación y
respuesta de sugerencias salen del loop con ownership nativo. Falta aplicar
estos recibos a su almacenamiento, presentación, vencimiento y participante de
borrado durable; este avance no cierra H3 ni habilita una release. La
[validación de recibos](../research/2026-10-26-h3-pattern-evidence-validation.json)
del commit `568d1fd` obtiene **4682 passed, 1 xfailed**, sin fallos, con 30 casos
nuevos, los checks y el artefacto compilado aceptados. La aceptación integrada
de runtime/GC y hardware continúa pendiente.

Las propuestas nuevas ya usan un almacén de política separado del JSON legacy.
Resuelven sus eventos reales al leer y al confirmar la escritura, conservan el
plazo anclado a la última observación y no lo renuevan al aceptar o posponer.
La voz interrumpe la entrega de una propuesta revocada, y la CLI comparte la
autoridad configurada y su lease. Aceptar entrega un borrador; la ejecución
efectiva sigue en H5. Falta conectar el mantenimiento físico y el participante
durable de sugerencias, además de los otros pendientes de H3 y del plan completo.
La [validación del almacén](../research/2026-10-26-h3-suggestion-store-validation.json)
del commit `3242ef5` registra **4721 passed, 1 xfailed**, sin fallos, con 39 casos
nuevos y el artefacto compilado aceptado. Incluye la corrección de un hueco al
reiniciar con menor retención, antes del primer barrido; la suite verde anterior
no lo había detectado. La aceptación integrada de GC, arranque y hardware sigue
pendiente.

El participante durable de sugerencias ya conecta sidecars actuales e históricos
y su runtime, también con analytics apagado. El borrado espera a los jobs y a los
workers de voz ya admitidos, conserva la cuarentena desconocida y revalida las
copias restauradas al reiniciar. Corrige además un enlace de traceback que podía
retener el resultado de un worker después de cancelar al observador. Quedan cinco
participantes; el mantenimiento físico independiente de TTL se conecta en el
avance siguiente. No hay un borrado global ni una release habilitada por esto.
La [validación del participante](../research/2026-10-26-h3-suggestion-erasure-validation.json)
del commit `ad2386f` registra **4750 passed, 1 xfailed**, sin fallos, con 29 casos
nuevos, los checks y el artefacto compilado aceptados. El ensayo incluye síntesis
y abortado nativos pendientes, reintentos de ack y limpieza con el pool general
saturado, sin ampliar límites. La aceptación integrada de runtime/GC y hardware
continúa pendiente.

El [mantenimiento independiente de propuestas](2026-10-26-h3-memory-privacy.md#avance-vencimiento-físico-independiente-de-propuestas)
limpia sidecars actuales e históricos aun con analytics apagado. Verifica
eventos reales en lotes acotados, conserva fuentes desconocidas y cuarentena,
persiste plazos menores y espera recursos nativos reales al cerrar. La
validación dirigida incluye vencimiento sin lectores, reinicio, 300 recibos,
fallos, fuentes bloqueadas y voz tardía. Siguen abiertos H3, la aceptación
integrada de runtime/GC y hardware, y el resto de H0–H6.
La [validación del vencimiento](../research/2026-10-26-h3-suggestion-retention-validation.json)
del commit `cba30ec` registra **4782 passed, 1 xfailed**, sin fallos, con 32
casos nuevos y el artefacto compilado aceptado. Incluye la corrección de una
carrera previa de capturas que podía perder el localizador de un clip al
revocar su conversación durante la purga; la primera suite fallida se conserva
como evidencia y no se acepta como verde.

**Entregables de privacidad:** texto y clips con TTL de 48 horas, hechos
automáticos con TTL de 30 días y recuerdos confirmados sin caducidad automática.
Registrar procedencia y fechas para poder borrar los derivados de una
conversación. Una consulta a memoria no renueva por sí sola la retención.

La voz puede proponer un recuerdo permanente; el dashboard autenticado lo
confirma o rechaza. Allí se autorizan también los borrados amplios. Separar pausa
de ambient de mute total: pausar ambient puede conservar wake y comandos; un
mute total necesita un control de reactivación que no dependa de oír una orden.

Inventariar los hechos existentes antes de migrarlos. No atribuir autor,
confirmación o fecha de origen a registros que no los tienen; resolver su
tratamiento mediante una vista previa de migración antes de modificar datos.

**Entregables de voces:** una vista autenticada muestra grupos de probable
hablante con ID estable, transcripciones, horarios, habitación, evidencia de
similitud y fragmentos reproducibles mientras estén vigentes. Permite nombrar
grupos, unirlos, separar capturas y dejar segmentos como desconocidos. Mostrar
por separado una sugerencia automática y una asociación confirmada manualmente;
una puntuación de similitud no se presenta como probabilidad calibrada de identidad.

Reutilizar `SpeakerTagger`, `SpeakerIdentifier` y `AmbientStore` donde alcance.
Hoy el tagger compara con perfiles registrados y devuelve `unknown` cuando no
hay perfiles: no existe todavía el agrupador de hablantes desconocidos ni su
vista de revisión. Evaluar agrupación con capturas representativas antes de
elegir o cambiar modelos. Procesar en background sin cambiar la asignación de
GPUs ni bloquear el camino de comandos.

**Salida:** los vencimientos se aplican también al consultar y después de un
reinicio. Borrar una conversación alcanza sus derivados y evita que un job
pendiente los recree. Pausa y reanudación tienen estado visible. La revisión de
voces conserva las correcciones manuales, respeta los TTL y permite abstenerse
ante audio corto, superpuesto o ambiguo. No agrupa todo `unknown` como una única
persona ni deduce identidad solo por habitación o texto.

La evaluación de agrupación debe incluir la misma persona a distintas
distancias y habitaciones, personas diferentes, TV, eco y voces superpuestas.
Separar calibración y evaluación, medir uniones erróneas y separaciones
erróneas y fijar el umbral de aceptación antes de evaluar. La vista y sus tests
no bastan para declarar correcta la agrupación acústica. Una asociación con un
nombre nunca otorga permisos ni cambia por sí sola qué se considera voz viva.

### H4 Recuperar dependencias y mostrar el estado local

**Entregables:** reconexión a HA y gateway, estados de disponibilidad por
subsistema, watchdog de audio integrado y cierre ordenado de tareas y recursos.
El dashboard muestra pérdida de captura y degradación; no se agrega un canal
externo de notificaciones en esta etapa.

**Salida:** HA o gateway tardíos se recuperan sin reiniciar manualmente KZA;
perder el cloud no anula el control local. Una captura sin frames produce
recuperación o fallo visible en el diagnóstico local. Validar cancelación,
SIGTERM y agotamiento de recursos en los escenarios definidos por el plan,
sin confundir el preflight de VRAM con recuperación de OOM. La última validación
con la unit real se realiza en H6.

### H5 Completar las funciones de voz incluidas

**Entregables:** conectar timers al router, crear y cancelar recordatorios por
voz y hacer que las operaciones de rutinas usen el scheduler persistente nuevo.
Conservar una única fuente de verdad entre dashboard y voz. Corregir la entrega
de recordatorios cuando no hay detector de presencia y definir su zona de salida.
Conectar las fuentes existentes de alertas a los estados de HA que estén
disponibles y comprobar su salida local en voz o dashboard según configuración.

Las operaciones disponibles por voz sobre rutinas y recordatorios deben heredar
la política de H1. No atribuir propiedad personal a una voz desconocida. El
diagnóstico de KZA se integra en H4; las alertas locales de producto se verifican
aquí. Postergar un canal externo para avisos de caída no elimina ese trabajo.
Música queda para un roadmap posterior.

**Salida:** pruebas del grafo construido por `main` verifican crear, consultar,
ejecutar y cancelar o borrar, según cada función. Se comprueba persistencia y
comportamiento tras reinicio. El resultado anunciado coincide con el registrado;
las acciones inciertas no se anuncian como confirmadas ni se reenvían a ciegas.
Un evento de HA simulado debe llegar a la alerta local correspondiente sin
convertir esa alerta en una acción de control fuera de la política común.

### H6 Integrar y publicar una sola release

**Entregables:** configuración validada para los subsistemas activos,
dependencias reproducibles de runtime, matriz de capacidades actualizada y
artefacto compilado del commit candidato. Resolver la configuración nocturna:
debe consumir los campos correctos y tener utilidad y recursos verificados; si
no, mantenerla deshabilitada en el perfil de release. No se desarrolla LoRA nuevo.

Cerrar los diez fallos del rescate y ejecutar los checks requeridos sobre el
candidato integrado. Medir latencia de punta a punta con límites explícitos y
p50/p95, falsos positivos y rechazos del wake, y calidad de agrupación de voces.
Registrar corpus, tamaños de muestra, revisión y entorno; no heredar como metas
cumplidas las cifras históricas del review.

**Salida para publicar:** H1 a H5 completos, suite y checks requeridos sin fallos
sin resolver, migraciones probadas en copia, restauración probada y artefacto
verificado fuera del checkout de fuentes. La prueba de rollback debe restaurar
un conjunto compatible de runtime y datos, además del enlace a una release.

La publicación sigue [DEPLOYMENT.md](../architecture/DEPLOYMENT.md): construir y
probar bajo `yo`; instalar solo artefactos y recursos de ejecución bajo `kza`.
Preparar el procedimiento y los respaldos antes de la intervención. En la
ventana de release, comprobar voz, controles seleccionados, frontend,
micrófonos, rutinas, memoria y revisión de voces. Promover únicamente si pasan
las comprobaciones; ante fallos, volver a la versión compatible anterior.
Retirar material de desarrollo de `kza` solo después de esa validación, según el
acuerdo existente. No hay deploys intermedios.

## Preguntas pendientes en el momento adecuado

Estas preguntas se resuelven con evidencia concreta durante H0 o al abrir el
plan correspondiente; no requieren pedir ahora al usuario IDs internos que
podemos inventariar.

| Pregunta | Cuándo se resuelve | Resultado necesario |
|---|---|---|
| Qué entidades y operaciones exactas se permiten, y qué escenas las agrupan | H0, mostrando el inventario al usuario antes de habilitarlas | Lista explícita y límites de clima y persianas; revisión de efectos de escenas. |
| Qué vigencia tienen las huellas y agrupaciones de voz, aparte de los clips | Plan de H3 | Política propia para embeddings, grupos automáticos y perfiles nombrados; los 48 h de audio no autorizan conservar huellas indefinidamente. |
| Qué datos previos no tienen procedencia y cómo se migran | H0 y plan de H3, con vista previa | Tratamiento de registros legacy y compatibilidad de rollback sin borrados masivos implícitos. |
| Qué actor y alcance tienen la creación por voz de rutinas y el borrado de la conversación activa | Planes de H1, H3 y H5 | Operaciones permitidas sin identidad, titularidad y alcance de borrado verificables. |
| Qué fuentes de alertas locales tienen datos reales y aportan valor | H0 y plan de H5 | Casos concretos de voz o dashboard con sus fuentes HA, evitando alertas sin productor o nuevas integraciones implícitas. |
| Qué niveles de latencia y calidad de agrupación son aceptables | Planes de H2 y H3, antes de medir el conjunto de evaluación | Objetivos medibles que no se eligen después de ver los resultados. |
| Qué ventana se usa para la única release | H6, con candidato y rollback probados | Intervención operativa concreta, sin reabrir usuario, directorio o modelo de despliegue ya acordados. |

## Fuera de alcance

Música, avisos externos de sordera o caída, nuevas integraciones para alertas,
cámaras, energía y entrenamiento nuevo de LoRA. Se conserva el
código existente mientras no haya una decisión de retirada. Tampoco se migra a
satélites de audio ni se cambia la asignación de GPUs dentro de este roadmap.

La clasificación de voces se usa para revisión y organización. No incorpora
autorización biométrica ni una promesa de identificar con certeza a todas las
personas. La elección de un proveedor cloud nuevo o de un canal externo de
avisos requiere su propio alcance posterior.
