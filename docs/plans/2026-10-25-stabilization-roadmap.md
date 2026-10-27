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
conservarlos en cuarentena. El ensayo privado de los 870 IDs ya pasó, con
documentos y vectores preservados y fuera de inferencia; faltan el corte real
y un conjunto completo de restauración compatible.
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
confirmación manual. El backend nuevo de huellas y perfiles revisados aplica
esa retención; agrupación, interfaz de revisión y migración siguen pendientes.

El avance del 27 de octubre agrega observación de voz en segundo plano y una
API de confirmación y borrado autenticados. Conserva el vencimiento original de
cada captura y perfiles manuales sin caducidad; las sugerencias de similitud no
identifican cuentas ni conceden permisos. El recibo local de `VOICES` espera la
salida real de modelos y escrituras. Los vectores legacy quedan excluidos y su
inventario mantiene bloqueado el borrado global. El alcance y las limitaciones
se detallan en el plan de H3. La [validación del registro](../research/2026-10-27-h3-voice-registry-validation.json)
pasó con 5.516 pruebas y un xfail conocido; el artefacto de `ad66e2b` verificó
344 hashes y la API real con encoder sintético. No cierra H3 ni habilita la
release.

El avance posterior agrega [muestras locales de revisión](../research/2026-10-27-h3-voice-capture-review-validation.json):
hasta seis segundos por huella, con transcripción del segmento, habitación y
vencimiento original, incluso si el perfil confirmado se conserva. La API
autenticada las lee sin vectores y revalida autoridad; siguen pendientes la
pantalla y sus recibos durables de retiro. La revisión `2adaca2` pasó 5.546
pruebas con un xfail conocido y verificó 345 hashes del artefacto. La evidencia
conserva tres fallos de la corrida inicial y la causa aún no demostrada de un
timeout de escritura. No cierra H3 ni la aceptación de H4/H6.

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

La [destilación vinculada a fuentes y trabajos reales](2026-10-26-h3-memory-privacy.md#avance-destilación-vinculada-a-fuentes-y-trabajos-reales)
ahora interrumpe inferencias al invalidarse el lote, conserva guardados async y
nativos hasta su salida real y mantiene la lease tras cancelar al observador.
Corrige también una retención de resultados tardíos desde el traceback del
monitor compartido. `PRODUCERS` sigue pendiente hasta cubrir STT, tagger,
buffers y comandos y conectar sus scopes al journal; este avance no cierra H3.
La [validación de destilación](../research/2026-10-26-h3-distillation-validation.json)
del commit `acee3be` registra **4812 passed, 1 xfailed**, sin fallos, con 30
casos nuevos, checks aceptados y el builder compilado ejercitado fuera del
checkout. Conserva evidencia de la retención de traceback corregida, sin
manipular GC ni ampliar límites; la aceptación integrada de H4/H6 sigue abierta.

Los [productores ambientales](2026-10-26-h3-memory-privacy.md#avance-productores-ambientales-con-seguimiento-nativo-y-retiro-de-buffers)
retienen STT, tagger, DoA y VAD reales tras cancelar; el retiro de preroll y
colas funciona sin otra entrada. El callback conserva su operación sin locks
bloqueantes y el cierre espera los jobs antes de soltar la lease. Quedan los
productores del command path, wake/sombra y su conexión completa al journal,
además de los otros pendientes de H3 y del alcance íntegro H0–H6.
La [validación ambiental](../research/2026-10-26-h3-ambient-producers-validation.json)
del commit `cdc27bf` registra **4848 passed, 1 xfailed**, sin fallos, con 36
casos nuevos, checks aceptados y scopes nativos ejercitados en el artefacto
compilado. Conserva la reproducción y corrección de una carrera del tap que
podía reinsertar audio retirado; la primera suite interrumpida no se acepta
como verde. Siguen abiertas la integración al journal y la aceptación física
y de runtime/GC del roadmap completo.

El procesador de comandos conserva ahora STT primario/sombra, lectura de
perfiles, identificación y emoción tras cancelar. Ambos routers le transmiten
las fuentes incluso sin memoria semántica; los resultados revocados o vencidos
no pasan al control. El timeout del veto limita la observación sin ocultar el
hilo real. La espera de modelo comprueba también cancelación, y el warmup del
pipeline conserva la misma autoridad hasta su salida efectiva.
La [validación de comandos](../research/2026-10-26-h3-command-producers-validation.json)
del commit `8ff3076` registra **4880 passed, 1 xfailed**, sin fallos, con 32 casos
nuevos, 625 pruebas dirigidas, checks y artefacto compilado aceptados. Conserva
las regresiones del helper paralelo y del aislamiento de un test de Piper y sus
correcciones. Quedan captura previa, wake/VAD, STT temprano, aprendizaje y journal
completo, los demás participantes y voces de H3 y el resto de H0–H6. No cierra
H3 ni la aceptación física o de runtime/GC.

La captura multiroom ahora mantiene wake, VAD, buffers, STT temprano y reset
en propietarios nativos por sala; el callback sólo conserva telemetría, tee y
entrada acotada. El evento conserva la procedencia creada antes de wake.
El retiro de un stream desacopla su cola y rechaza callbacks antiguos, incluso
si avanzó el reloj. La [validación de captura](../research/2026-10-26-h3-capture-producers-validation.json)
del commit `71c0588` registra **4908 passed, 1 xfailed**, 28 casos nuevos, 814
pruebas dirigidas y checks/artefacto compilado aceptados. Conserva la reproducción
del callback retirado y del helper que cancelaba un cleanup ajeno, y sus
correcciones. Quedan consumidores de clips y aprendizaje, journal completo,
los demás participantes, voces y el resto de H0–H6. No habilita borrado global
ni cierra aceptación física o runtime/GC.

El camino alternativo `AudioLoop`/`AudioManager` comparte ahora el actor de
captura y la autoridad de privacidad. El driver conserva apertura y cierre
nativos reales, incluso si el observador cancela o el cierre requiere reintentos.
Comandos y análisis ambiental avanzan con admisión acotada sin detener la captura;
las ventanas mezcladas conservan todas sus fuentes y el vencimiento más antiguo.
El detector opcional retiene carga, inferencia y callbacks reales y libera la
onda antes de notificar. La [validación de entrada única](../research/2026-10-26-h3-single-input-validation.json)
del commit `03c2f08` registra **4941 passed, 1 xfailed**, 33 casos nuevos,
702 pruebas dirigidas, checks y artefacto sin fuentes aceptados. Conserva dos
timeouts de preparación de eventos de la primera suite y la regresión que
comprueba escritura real y borrado sin cambiar el límite de 100 ms del
observador. Esto cubre ambos caminos de captura, con los alcances pendientes
anteriores; todavía no completa `PRODUCERS` ni `STREAMS`. Las pausas de GC y la
aceptación física e integrada siguen pendientes.

Los clips de wake aceptados, rechazados y missed conservan ahora sus fuentes y
vencimiento en el catálogo privado. Un intento durable precede al archivo; el
rename comparte la barrera de admisión y el cierre espera al writer real. La
retención funciona con la captura apagada, conserva raíces históricas y purga
la cola sin esperar a un encoder bloqueado. Los WAV legacy no se importan ni
rotan con los nuevos. La [validación de clips](../research/2026-10-26-h3-wake-clips-validation.json)
del commit `a239fa7` registra **4971 passed, 1 xfailed**, 30 casos nuevos,
checks y artefacto sin fuentes aceptados. Siguen pendientes los demás
colectores y datasets de aprendizaje, journal completo, pausa/mute, voces,
migración y el resto de H0–H6; no se anuncia borrado global.

Las correcciones STT comparten ahora los localizadores y la autoridad con los
clips. Audio original y feedback deben aportar fuentes; WAV y texto vencen con
el plazo más antiguo. Las vistas ya entregadas se revocan, y una cola vencida
se purga aunque el barrido físico y los encoders sigan ocupados. El plazo
durable acortado se comprueba antes y después de publicar y se propaga entre
colectores del mismo destino. `main` construye el colector configurado;
el entrenador Whisper antiguo queda bloqueado para esos datos privados antes
de escribir datasets o lanzar procesos. Esto no agrega un productor de
correcciones ni entrenamiento nuevo. Siguen pendientes conversaciones, otros
datasets y consumidores, journal completo y el resto de H0–H6.

La [validación de correcciones STT](../research/2026-10-26-h3-stt-corrections-validation.json)
del commit `042ebbb` registra **4998 passed, 1 xfailed**, 27 casos nuevos,
358 pruebas dirigidas y checks/artefacto sin fuentes aceptados. Una primera
suite verde no cubrió plazos acortados durante encoder/rename ni una vista
retenida por otro colector: la evidencia la rechaza y conserva las tres
reproducciones y la corrección compartida. No se ampliaron límites ni se
manipuló GC. Las pausas globales observadas antes y la aceptación física e
integrada siguen pendientes; tampoco se anuncia borrado global.

Los turnos nuevos del colector y sus exportaciones conservan ahora fuentes y
plazos. El feedback crea otra versión y retira la anterior; los exports guardan
sus padres para invalidarse al corregir o retirar un turno, también después de
reiniciar. El CLI usa el catálogo y genera lotes temporales con continuación;
los consumidores privados antiguos quedan bloqueados antes de copiar datos.
El perfil candidato deshabilita el entrenamiento nocturno no verificado; la
configuración instalada se conserva hasta H6.

La [validación de conversaciones](../research/2026-10-26-h3-conversation-collector-validation.json)
del commit `fc04490` registra **5034 passed, 1 xfailed**, 36 casos nuevos,
815 pruebas dirigidas y checks/artefacto sin fuentes aceptados. Conserva las
reproducciones de plazo vencido durante callback y commit de SQLite, y verifica
feedback por el router y retirada de exports tras corregir y reiniciar. No se
habilita un productor automático de turnos ni entrenamiento nuevo. Siguen
pendientes otros datasets y consumidores, journal completo, diagnósticos,
streams, voces, migración, validación física y el resto de H0–H6; no equivale
todavía a borrado global de `COLLECTORS` ni cierra H3.

La [admisión del entrenamiento antiguo](2026-10-26-h3-memory-privacy.md#avance-admisión-del-entrenamiento-antiguo)
también bloquea las lecturas de perfiles e historial en sus constructores,
antes de tocar disco. `main` pasa la autoridad privada: aun si se habilita el
flag nocturno, no inicia el scheduler ni anuncia que quedó programado. La
ejecución manual devuelve `blocked` sin datasets, modelos, callbacks o reportes.
La [validación](../research/2026-10-26-h3-private-training-admission-validation.json)
del commit `b793903` registra **5050 passed, 1 xfailed**, 16 casos nuevos,
129 pruebas dirigidas y checks/artefacto sin fuentes aceptados. Los archivos
legacy permanecen intactos; su migración/retención y los consumidores alternos
siguen pendientes. El CLI nocturno se corrige en el avance siguiente. Tampoco cierra
`COLLECTORS`, la aceptación integrada de runtime/GC, H3 o H6. Producción no cambia.

El [CLI nocturno y la entrada directa de LoRA](2026-10-26-h3-memory-privacy.md#avance-cli-nocturno-y-entrada-directa-de-lora)
ya no permiten acceder al entrenamiento o adapters sin una sesión admitida,
incluso al copiar o enlazar un export fuera de su directorio original. El
status distingue disponibilidad del CLI de un estado del runtime que no consultó;
las operaciones bloqueadas salen con código 1. Las estadísticas y exports del
catálogo siguen disponibles. No se agrega entrenamiento ni se cambian GPUs.

La suite inicial descubrió además un [bucle de cierre de memoria](2026-10-26-h3-memory-privacy.md#avance-cierre-de-memoria-con-callbacks-pendientes)
que impedía retirar tareas ya terminadas. Una reproducción determinista confirmó
el cruce de callbacks; la corrección conserva la propiedad de workers nativos
reales y su lease. La [validación conjunta](../research/2026-10-27-h3-training-cli-validation.json)
del commit `4f82a85` registra **5073 passed, 1 xfailed**, 23 casos nuevos,
checks y artefacto sin fuentes aceptados. Conserva la ejecución interrumpida
como no aceptada. El artefacto verifica 317 hashes, cuatro casos de cierre y
siete procesos CLI. No cierra la retención de datasets legacy, el journal
completo, la aceptación integrada de GC/hardware ni los demás hitos de H0–H6.

El [ensayo de memoria legacy en copia](2026-10-26-h3-memory-privacy.md#avance-ensayo-privado-de-la-selección-legacy-en-copia)
ya conserva exactamente los 870 IDs autorizados con documentos, metadata y
vectores. La colección tiene ahora 874 registros; los cuatro nuevos no entran
en esa importación. La copia es privada y el origen se abre sólo para lectura.
Reanudar no duplica ni recrea filas retiradas, y dos arranques del grafo de memoria
mantienen cero hechos, fuentes, propuestas, búsquedas y jobs a partir de ellos.
La [validación](../research/2026-10-27-h3-legacy-quarantine-copy-validation.json)
del commit `b8b3cdd` registra **5099 passed, 1 xfailed**, 26 casos nuevos,
checks y artefacto sin fuentes aceptados, con 319 hashes verificados. Los datos
reales quedan fuera de Git y del artefacto. No se migra producción ni se afirma
un rollback completo: faltan HNSW/restauración compatible, revisión manual,
demás migraciones, ciclo de vida de snapshots/cuarentenas y los otros entregables.

El [journal de colectores](2026-10-26-h3-memory-privacy.md#avance-journal-de-archivos-y-scopes-nativos-de-colectores)
ya tiene un driver conectado en `main` para wake, correcciones STT, conversaciones
y exports. Los recibos esperan la salida nativa real, conservan los locators
hasta completar el borrado físico y vuelven a comprobar los destinos históricos
tras reiniciar. Un disco bloqueado no impide revocar otra conversación. Los
archivos legacy, cuarentenas y snapshots sin procedencia verificable continúan
preservados y pendientes. Este avance no completa el borrado global, los demás
participantes, la aceptación integrada de GC/hardware ni H3–H6.
La [validación](../research/2026-10-27-h3-collector-journal-validation.json) del
commit `780f212` registra **5137 passed, 1 xfailed**, 38 casos nuevos, checks
aceptados y artefacto sin fuentes con 320 hashes verificados. No se leen datos
reales del hogar ni se modifica la copia privada anterior o producción.

La revisión de productores detectó y corrigió una [reintroducción de audio tras
rotar la conversación](2026-10-26-h3-memory-privacy.md#avance-impedir-que-una-rotación-renombre-audio-atrasado).
El límite de admisión y de buffers avanza por habitación y origen; un paquete
anterior al intercambio vigente no puede recibir otro ID. Los paquetes atrasados
del mismo intercambio conservan su ID sin retroceder la fecha de actividad.
Las pruebas incluyen tap, cola de comandos y VAD retenido reales. Sigue pendiente
la conexión completa de `PRODUCERS`, incluidos los datos entre handoffs y las
lecturas de destilación anteriores a conocer su procedencia.
La [validación](../research/2026-10-27-h3-capture-scope-rotation-validation.json)
del commit `864c2ed` registra **5160 passed, 1 xfailed**, 23 casos nuevos,
checks aceptados y artefacto sin fuentes con 320 hashes verificados. No completa
`PRODUCERS`, H3 ni la aceptación de hardware o de la release.

Los [handoffs ambientales y las lecturas de destilación](2026-10-26-h3-memory-privacy.md#avance-scopes-finitos-de-handoffs-ambientales-y-lecturas-de-destilación)
ya tienen scopes de metadata que siguen a los hijos reales tras cancelar al
padre. Las lecturas anteriores a conocer su procedencia permanecen retenidas,
incluido el thread SQLite; el worker ambiental de larga duración deja de guardar
arrays entre awaits. La [validación](../research/2026-10-27-h3-producer-scope-handoffs-validation.json)
del commit `063b444` registra **5179 passed, 1 xfailed**, 19 casos nuevos,
checks aceptados y artefacto sin fuentes con 321 hashes verificados. Falta cubrir
los buffers en reposo y captura/comandos y conectar los contratos completos al
driver de `PRODUCERS`. No se publica ni se acepta hardware con este paso.

La [afinidad de comandos y controladores de captura](2026-10-26-h3-memory-privacy.md#avance-afinidad-finita-de-comandos-y-controladores-de-captura)
ya incluye argumentos que esperan el lock, sombras, emoción directa y llamadas
nativas tras cancelar al observador. Las llamadas auxiliares conservan las
fuentes del llamador y no devuelven resultados revocados; sus errores tipados
evitan transportar tracebacks con argumentos crudos. La
[validación](../research/2026-10-27-h3-command-capture-scopes-validation.json)
del commit `67e589e` registra **5203 passed, 1 xfailed**, 24 casos nuevos,
checks aceptados y artefacto sin fuentes con 321 hashes verificados. Faltan los
handoffs de los loops receptores, buffers en reposo, callbacks de entrada y el
driver completo de `PRODUCERS`. Producción sigue intacta.

Los [receptores de captura](2026-10-26-h3-memory-privacy.md#avance-recibos-de-entrega-a-los-loops-receptores)
ya conservan la fuente antes de recibir el resultado del kernel y procesan
ciclos finitos en entrada única y multiroom. También se impide el postprocesado
tardío tras revocación y se vacían los argumentos auxiliares antes de que el
executor publique su resultado. La
[validación](../research/2026-10-27-h3-capture-receiver-scopes-validation.json)
del commit `3c779e2` registra **5227 passed, 1 xfailed**, 24 casos nuevos,
checks aceptados y artefacto sin fuentes con 321 hashes verificados. Conserva
como no aceptada la suite anterior a los dos fallos descubiertos después.
Las [lecturas de memoria](../research/2026-10-27-h3-memory-prompt-read-validation.json)
y la [cola independiente](../research/2026-10-27-h3-queue-receiver-lineage-validation.json)
ya publican las dependencias seleccionadas al receptor antes de entregar el
contexto y guardan los efectos posteriores de respuestas derivadas.
Los [callbacks y buffers locales](../research/2026-10-27-h3-input-callback-retirement-validation.json)
conservan ahora sus invocaciones reales, barreras y estado en reposo hasta
retirarlos. Un reset fallido conserva procedencia y autoridad para reintentar.
La validación de `2f77ead` registra **5284 passed, 1 xfailed**, 21 casos nuevos,
183 pruebas dirigidas, checks y un artefacto sin fuentes con 322 hashes.
El [estado recurrente ambiental](../research/2026-10-27-h3-ambient-reset-retirement-validation.json)
también conserva el segmentador, la procedencia y el lease ante fallos de
reset. Los segmentos trasladan esa procedencia a la transcripción sin renovar
su vencimiento. La validación de `a47385d` registra **5300 passed, 1 xfailed**,
16 casos nuevos, 288 pruebas dirigidas, checks y artefacto con 322 hashes.
Quedan la coordinación entre orígenes, los demás productores, los buffers de
backends/SDK y el driver completo de `PRODUCERS`, además del resto de H3–H6.
Producción permanece intacta; la aceptación física e integrada sigue pendiente.

El [ingreso compartido de voz y ambient](../research/2026-10-27-h3-shared-input-retirement-validation.json)
conserva ahora sus entregas por separado: cerrar ambient espera su copia real
y sus `push`, sin detener ni esperar la copia posterior de voz. La barrera
ambiental publicada rechaza entregas anteriores aunque voz tenga otra
conversación. La validación de `778c798` registra **5313 passed, 1 xfailed**,
13 casos nuevos, 422 pruebas dirigidas, checks y artefacto con 322 hashes.
Faltan el intervalo entre revocación y publicación, la procedencia por paquete
en las demás colas y el contrato completo de `PRODUCERS`. Esto no completa
la pausa/mute de producto, las pruebas físicas ni el resto del roadmap.

Las [marcas de admisión](../research/2026-10-27-h3-capture-ingress-epochs-validation.json)
ya viajan desde antes del reloj del callback hasta las colas y los consumidores
nativos. El ledger las retira antes del commit de revocación, de modo que un
audio anterior no adquiere otra conversación por recibir una fecha posterior;
una entrada fresca puede continuar incluso con el commit pendiente. La
validación de `5bd8da9` registra **5338 passed, 1 xfailed**, 25 casos nuevos,
601 pruebas dirigidas, checks y artefacto con 322 hashes. Conserva la suite
previa de seis fallos como no aceptada. La calidad de audio al rotar marcas,
los buffers de drivers/SDK, los demás productores y el contrato global del
journal siguen pendientes, junto con el resto de H3–H6. Producción no cambia.

La [poda en reposo](../research/2026-10-27-h3-idle-ingress-retirement-validation.json)
ya retira paquetes y segmentos con marcas revocadas, vencidas o rotadas aunque
su fecha aparente seguir vigente. Cambiar de marca desacopla la cola incluso
con la misma fecha de corte; las copias aún en curso siguen pendientes hasta
salir. El segmento conserva la marca causal después del reset nativo y se
comprueba antes de STT. La validación de `c0f984e` registra **5359 passed,
1 xfailed**, 21 casos nuevos, 511 pruebas dirigidas, checks y artefacto con 322
hashes. La otra sala conserva su cola válida. No se completa el contrato global
del journal, los SDK, las voces, la aceptación física ni el resto del roadmap.

El [diagnóstico numérico](../research/2026-10-27-h3-numeric-diagnostics-validation.json)
ya excluye payloads de voz e identidades en los productores de métricas y en
sus copias JSONL/TCP. El I/O corre fuera del loop con reservas acotadas y cierre
retenido; los posts no capturan contexto libre. Los logs propios de los tres
adapters STT y de esos puntos de latencia/wake conservan sólo diagnóstico
acotado. La validación de `4806520` registra **5372 passed, 1 xfailed**, 13 casos
nuevos, 250 pruebas dirigidas, checks y artefacto con 322 hashes. El JSONL nuevo
usa otro nombre y no reenvía archivos anteriores. Siguen pendientes la
procedencia/retención completas, las copias legacy y remotas, otros logs y el
participante `DIAGNOSTICS`; no se cierra H3 ni se modifica producción.

La [retención local del diagnóstico](../research/2026-10-27-h3-diagnostic-retention-validation.json)
añade fechas de admisión y vencimiento persistentes, poda en reposo y reparación
del JSONL activo y rotado desde un catálogo propio. Acortar el plazo queda
registrado antes de publicar; reiniciar con un plazo mayor no restaura filas
borradas desde una copia vieja. La importación del esquema numérico anterior
conserva su fecha original; archivos no verificables quedan en cuarentena para
revisión manual. Los originales sin versión siguen intactos. Un lease conserva
la propiedad hasta terminar el trabajo nativo real. La validación corresponde
a `3b18d0c`, con **5387 passed, 1 xfailed**, 15 casos nuevos, 265 pruebas
dirigidas, checks y artefacto sin fuentes de 323 hashes. La poda
requiere proceso e I/O disponibles; no se promete eliminación física durante
una interrupción. Siguen pendientes procedencia por conversación, borrado
remoto, otros logs y el participante `DIAGNOSTICS`. No se cierra H3.

El [borrado del diagnóstico numérico](../research/2026-10-27-h3-diagnostic-erasure-validation.json)
vincula las métricas nuevas con conversaciones en el catálogo privado, sin
publicar ese vínculo en JSONL/TCP. El driver del journal limpia archivos activos
e históricos y espera los escritores reales de la conversación. Los históricos
comparten un worker; siguen manteniéndose con emisión y memoria deshabilitadas.
Las filas sin procedencia, cuarentenas y fallos físicos impiden completar el
recibo local. Un reinicio reabre recibos para comprobar restauraciones. El
inventario restante de diagnósticos queda explícitamente bloqueado: falta
resolver otros logs, originales sin versión y copias remotas. La validación
corresponde a `942210f`: **5404 passed, 1 xfailed** en la suite final, 17 casos
nuevos, 289 pruebas dirigidas, checks y artefacto con 324 hashes. Conserva un
ensayo completo anterior no aceptado por pausa del loop de 92,90 ms sobre el
límite de 75 ms. Las mediciones posteriores pasaron sin cambiar código ni
límites, pero no explican ese pico. La latencia integrada de H4/H6 sigue abierta;
este paso no completa `DIAGNOSTICS` ni H3.

El [canal Live](../research/2026-10-27-h3-live-stream-validation.json) ya exige
fuentes y autoridad, copia una proyección acotada y retira registros en reposo.
Conserva la entrega y cierre reales del WebSocket después de cancelar o agotar
la espera del observador. La sesión se revalida también en conexiones quietas;
la UI limpia el ticker por revocación, vencimiento, desconexión o pérdida de
sesión. Chromium verifica esos cambios sobre el DOM real. La evidencia de
`40080e0` registra **5424 passed, 1 xfailed**, 20 casos nuevos, 403 pruebas
dirigidas, 23 casos de frontend, checks y artefacto de 326 hashes. Conserva la
fuga de contexto en logs detectada y corregida durante la validación y su suite
parcial detenida. Quedan el ACK y recibo durable del navegador, otros WebSockets y
caches de otras vistas; `STREAMS`, H3 y la aceptación integrada siguen abiertos.

Los [recibos del navegador de Live](../research/2026-10-27-h3-live-browser-receipts-validation.json)
ya persisten tickets sin texto antes de enviar, exigen un reset confirmado al
reconectar y esperan la retirada de caché y DOM. React conserva sólo el
contenedor; Chromium verifica que el ACK no salga antes de confirmar la
retirada. El driver local del journal espera también el callback real del
servidor y reabre recibos al restaurar metadatos. La revisión validada en
`14700cc` corrige además la herencia de contexto privado en tareas y callbacks
de finalización del coordinador. Pasaron 5.444 pruebas con 1 xfail, 465 pruebas
focalizadas, 30 de frontend, Chromium y el artefacto compilado con 329 hashes
verificados. El inventario restante de `STREAMS` sigue bloqueado: otros
WebSockets, caches REST, clientes legacy y páginas sin confirmación necesitan
cobertura. No se cierra H3.

La [vista Conversaciones por REST](../research/2026-10-27-h3-conversation-deadlines-validation.json)
ya propaga el vencimiento original de todas sus capturas y retira la caché,
el DOM y el filtro de voz al vencer. Sus textos dejaron de poblar los mocks y
el estado de React; los IDs provienen del registro persistido y distinguen
acciones simultáneas de una misma captura. Esta cobertura local de TTL no
resuelve los recibos REST, la invalidación anticipada por borrado ni las cachés
de las demás vistas. `STREAMS` y H3 siguen abiertos.

Los [recibos REST de Conversaciones](../research/2026-10-27-h3-rest-receipts-validation.json)
ya ligan cada lectura real a una conexión autenticada y un reset confirmado,
persisten un ticket antes de enviar y retiran el lote al revocarse sus fuentes.
El journal espera tanto la confirmación de caché y DOM como las lecturas nativas,
buffers del middleware y callbacks exteriores realmente activos. La cancelación
del observador no libera esas operaciones. Su destino local versión 3 conserva
el bloqueo explícito del inventario restante: otras vistas, WebSockets, clientes
legacy y páginas sin confirmar impiden cerrar `STREAMS` y H3.
La suite completa de esta revisión no se acepta: 5.453 pruebas pasaron y cuatro
fallaron por límites de inicialización o cierre de recursos del colector y de
diagnóstico, con 1 xfail. Los cuatro casos pasaron aislados sin cambiar límites;
esa repetición no explica los fallos ni cierra la aceptación de H4. También se
conserva un arranque de navegador que excedió su plazo, aunque el ensayo
diagnóstico posterior pasó con el mismo límite.

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

La [observación de cierres nativos y arranque local](../research/2026-10-27-runtime-close-offline-validation.json)
ya distingue fases y esperas sin retener datos privados. Se corrigió una
adquisición de lock desde el loop al repetir el cierre de la autoridad, y
tareas de cierre que conservaban contexto del solicitante. El dashboard sirve
sus bibliotecas ya fijadas desde el propio artefacto, con hashes y licencias,
y tiene una comprobación de navegador con el CDN bloqueado. Los timeouts
históricos conservan su evidencia: estos cambios no prueban por sí solos su
causa ni completan la aceptación de H4.
La revisión `da88855` pasó 5.474 pruebas con 1 xfail, 49 de frontend y Chromium
con assets externos bloqueados. El artefacto verificó 340 hashes. Los cuatro
timeouts previos no reaparecieron en esta suite; la aceptación integrada de
recursos, GC, dependencias y hardware sigue pendiente.

El [cierre del logger de eventos](../research/2026-10-27-h3-voice-capture-review-validation.json)
ya observa la salida real de su pool privado sin depender del executor general
y sin copiar contexto del llamador. Sus fases y tiempos nativos son acotados y
sin contenido. Dos casos rojos reprodujeron ambas condiciones; la suite final
pasó 5.546 pruebas con un xfail. Los plazos permanecen iguales. Esto no demuestra
la causa de todos los timeouts observados ni resuelve la aceptación de GC,
capacidad o latencia.

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
