# H3: procedencia, retención y revisión de memoria

Este avance implementa el contrato de memoria del
[roadmap](2026-10-25-stabilization-roadmap.md) sin publicar una release.
La retención previa de capturas está documentada en la
[validación de capturas](../research/2026-10-26-h3-capture-retention-validation.json).
No cierra H3: quedan los borrados coordinados, las otras copias de texto y audio,
pausa/mute y la revisión de voces.

## Decisiones y alcance

- Texto y clips: hasta 48 horas desde la captura, sin renovación al leer.
- Hechos automáticos: hasta 30 días desde extracción, con fuentes identificadas.
- Propuestas: fuera de la inferencia; vencen con la fuente más temprana, hasta
  48 horas. Una orden de voz puede proponer, pero no confirmar permanentemente.
- Recuerdos confirmados: sin vencimiento automático; confirmación autenticada,
  con fecha y actor. La etiqueta de SpeakerID no concede esa autoridad.
- Los 870 hechos legacy sin fuentes verificables quedan en cuarentena hasta
  revisión manual, conforme a la decisión del usuario del 26 de octubre.
  El JSON previo de preferencias tampoco prueba confirmación autenticada.

El desarrollo conserva los originales legacy y evita usarlos en respuestas.
Todavía falta ensayar la migración sobre una copia consistente de datos reales,
incluida la restauración de datos y runtime compatibles. No se atribuye a la
inserción de Chroma la fecha de una conversación ni se inventan fuentes.

## Autoridad y reconciliación

`FactLedger` es el catálogo SQLite que autoriza cada lectura y transición.
Guarda contenido, estado, extracción, vencimiento y revisión, fuentes con UUID
de captura y conversación, y procedencia de la confirmación. El archivo es
privado y se abre desde workers del servicio de memoria.

Chroma es un índice reconstruible separado de la colección legacy. Sus
documentos y metadatos no autorizan retención ni confirmación. La búsqueda se
restringe a IDs vigentes del catálogo y vuelve a consultar cada candidato antes
de devolver contenido. Una entrada vencida, borrada o pendiente de confirmar
no aparece aunque la limpieza del índice falle.

Cada mutación relevante encola una revisión en la misma transacción SQLite.
La reconciliación procesa lotes acotados y sólo confirma la revisión que escribió:
un upsert lento no puede dar por cumplida una eliminación posterior. Los fallos
de índice conservan trabajos para reintento, también tras reinicio. No se usa un
reseteo global del cliente Chroma para recuperar una operación.

El ID determinista de derivación evita duplicar o renovar un hecho al reintentar
el mismo lote de fuentes. El distiller sólo marca el lote completado cuando todos
sus hechos quedaron guardados. Conservadoramente, cada hecho enlaza todas las
capturas retenidas del lote; todavía no se afirma atribución por frase.

Un tombstone de conversación impide nuevas derivaciones antes de eliminar los
hechos existentes. El guardado vuelve a validar fuentes y tombstones después de
la extracción. Esta garantía corresponde al catálogo: aún no se expone una
operación HTTP que prometa borrar una conversación de todos los almacenes.

## Integración y revisión

`build_privacy_services` inicializa siempre la autoridad y su lease, antes de
la memoria opcional. `build_memory_services` registra su cierre antes de
inicializar el índice y toma prestado ese mismo catálogo. La memoria mantiene
su reconciliación; la autoridad mantiene vencimientos incluso con la memoria
semántica apagada.
Los workers nativos son acotados y siguen registrados si se cancela al llamador.
El cierre informa recursos retenidos si no terminan dentro de su presupuesto;
la lease y SQLite permanecen abiertos hasta terminar sus usuarios reales.

Main, ambos routers y el distiller comparten este servicio. Las capturas nuevas
conservan ID, tiempo y conversación desde su productor; wake textual conserva
los IDs ambientales. Los scopes por habitación y pausas describen intercambios
grabados, sin atribuir persona ni titularidad. El contexto corto respeta el menor
vencimiento de sus fuentes, con máximo de 48 horas.

La pestaña Memoria consulta un snapshot autenticado con reloj del servidor.
Permite conservar una propuesta sin vencimiento, descartarla o eliminar un
recuerdo vigente. Una propuesta vencida no se confirma, y rechazar desde una
vista desactualizada no elimina una confirmación realizada entretanto.
Las respuestas privadas tienen `Cache-Control: no-store`. El cliente elimina
filas vencidas, descarta respuestas tardías y limpia datos del DOM al perder la
sesión o cambiar de vista. Un resultado desconocido obliga a consultar el estado
antes de actuar nuevamente.

## Contexto y resúmenes del orquestador

El avance posterior propaga las mismas fuentes a los dos turnos de cada
intercambio del orquestador y al resumen de compacción. El resumen y sus IDs
referenciados vencen con la fuente más temprana; ni compactar nuevamente,
guardar un snapshot, consultar ni rehidratar una sesión renuevan ese plazo.
Las proyecciones de texto, chat, entidades y serialización descartan fuentes
vencidas o revocadas. El contexto corto del MemoryManager usa esa misma política.

La escritura de un snapshot valida fuentes mientras mantiene la transacción
que ordena su commit respecto de los tombstones. Una escritura anterior al
marker ya no es legible después de éste; una escritura que llega después se
rechaza. Los snapshots nuevos privados viven bajo `contexts/policy_v3`, con
procedencia de los prompts explícita. Los resúmenes anteriores de `policy_v2`
no se rehidratan como independientes: se conservan fuera de inferencia hasta
su vencimiento original de captura. Los JSON anteriores a esa política siguen
fuera de esta lectura, sin reescribirlos. El worker elimina snapshots conocidos vencidos o revocados y
temporales que no se promovieron, también al arrancar.

La compacción comprueba identidad y revisión del contexto antes de publicar,
y vuelve a validar sus fuentes tras esperar al reasoner. Limpiar o reemplazar
un contexto invalida su resultado tardío. El snapshot final conserva un contexto
que volvió a activarse mientras esperaba. La carga y escritura de disco se
ejecutan fuera del loop, sin mantener el lock del contexto durante I/O. Se
registran los trabajos nativos incluso si se cancela al llamador; el cierre
espera su salida real o comunica el vencimiento de su presupuesto.
Un contexto ya cargado se consulta sin ocupar otro worker, incluso con las ocho
operaciones nativas en curso. La reactivación se detecta mediante una revisión
propia de actividad, sin depender de que cambie el reloj de pared.

Esta integración no expone todavía un borrado amplio de conversación: faltan
las capturas/clips y otras copias, sus productores y la coordinación durable
desde el dashboard. Tampoco transforma los perfiles de usuario ni las
preferencias legacy en confirmaciones manuales.

## Eventos de diagnóstico y actividad

El registro nuevo usa la tabla `events_policy_v1` en el archivo configurado de
analítica. La tabla legacy se conserva sin importar ni modificar sus filas.
Todas las consultas del logger —historial, distribución horaria/semanal,
secuencias y estadísticas— usan una proyección que excluye eventos revocados
o vencidos y aplica el vencimiento de texto al leer.

La frase de disparo, el nombre copiado y el contexto libre conservan el menor
vencimiento de sus capturas, hasta 48 horas. Los metadatos de actividad y las
señales numéricas siguen el plazo configurado en `analytics.retention_days`,
actualmente 90 días; ese deadline se persiste por evento y no se renueva al
consultar ni al cambiar la configuración después. Los hechos del catálogo de
memoria conservan su política propia de 30 días. El logger admite en el canal
numérico sólo campos y tipos conocidos, para que el contexto libre no prolongue
la transcripción. Una escritura sin fuentes guarda exclusivamente ese
diagnóstico numérico y los identificadores de actividad validados.

Las dos rutas de voz y sus resultados de aceptación o rechazo adjuntan la
procedencia del CommandEvent. También existe esa procedencia al deshabilitar
el catálogo de memoria; no se toma el momento de terminar la métrica como
inicio de la captura. Antes de guardar texto se revalidan fuentes y tombstones
bajo la transacción de la política. Un writer anterior termina antes del marker;
uno que llega después se rechaza. Las consultas filtran la revocación aunque
todavía falte la eliminación física de la fila.

El factory inicializa el registro fuera del loop y registra su cierre antes de
iniciar I/O. El mantenimiento arranca con limpieza y se despierta según el
próximo plazo; elimina eventos vencidos o revocados y retira el texto vencido
de los que aún conservan métricas. Las conexiones SQLite se cierran al salir
de cada operación. `OwnedBlockingWork` limita los trabajos nativos y conserva
su ownership tras cancelación o timeout, también cuando delega en el worker
compartido de memoria.

La espera de escritura durante voz está acotada por
`analytics.operation_timeout_s`; el valor inicial es 0,1 s. Fallar una métrica
no revierte una acción ni convierte su diagnóstico en otro intento de HA. Las
lecturas de eventos de la API autenticada también se ejecutan fuera del loop;
sus respuestas y errores llevan `Cache-Control: no-store`.

Quedan fuera del registro SQL los demás logs de diagnóstico, colectores y copias
mantenidas por consumidores del stream o del frontend.
La revisión de sugerencias derivadas y su procedencia se integra con el trabajo
restante de H3/H5. El borrado coordinado continúa pendiente.

## Solicitudes e índices transitorios

Las solicitudes nuevas conservan fuentes desde su admisión. Si se ingresan
directamente por la API interna de cola, el intercambio se registra como
`interaction`; una captura de voz conserva su UUID y plazo originales.
Las proyecciones de cola y la publicación de resultados consultan vencimiento
y revocaciones. Un pedido inválido pendiente o activo se cancela con motivo de
privacidad; se retiran texto, nombre copiado, resultado, errores y closures.
Los callbacks también se liberan al llegar a un estado terminal, y un listener
registrado sobre un token ya cancelado se ejecuta fuera de su lock.

Los recibos completados se conservan sólo mientras sus fuentes están vigentes.
La limpieza elimina esos cuerpos del tracking por usuario y de la cola; el
puntero del trabajo actualmente en cleanup conserva su ownership hasta la salida
real. Una cancelación no presenta ese trabajo como terminado. El cierre del
orquestador descarta los cuerpos que todavía se retenían.

`TransientCache` limita la caché de consultas y conserva las fuentes y su plazo,
sin renovar por hits. Claves y valores desaparecen al vencer o revocarse; las
copias defensivas impiden que un consumidor modifique el valor retenido. Cerrar
la caché impide que un resultado tardío vuelva a poblarla.

El historial del clasificador usa el menor plazo entre las capturas y la ventana
configurada de NLU (120 s por defecto). La ventana parte del tiempo de captura,
no del momento de completar la llamada. El resultado del modelo depende tanto
de la captura actual como de las entradas históricas incluidas en el prompt:
se comprueban antes de iniciar la operación nativa y después de su retorno.
Ambas rutas registran una vez los controles aceptados, incluidos los que terminan
en el camino tipado, con las fuentes reales de ese intercambio.

El router mantiene una tarea propia que purga estas copias en inactividad con
intervalo nominal de un segundo. Las lecturas aplican el plazo sin esperar ese
tick. La inicialización y el cierre se conectan a VoicePipeline. Clasificación
nativa, búsqueda vectorial y la interpretación rápida del router tienen trabajo
acotado y retenido tras cancelación o timeout. Se cierra la admisión antes de
esperar su salida; un lookup tardío no vuelve a llenar la caché ni envía una
acción desde una captura invalidada. Se retiran los campos de texto del resultado
que ya no puede publicarse, conservando los recibos de efectos ya efectuados.

Siguen pendientes las copias en otros logs, colectores, streams y frontend, y
la coordinación durable de todos los almacenes. El seguimiento transitivo de
los prompts de las rutas activas se incorpora en el avance siguiente; todavía
no equivale a una auditoría completa de todos los productores. El corpus y la
evaluación acústica de voces permanecen separados de estas pruebas de software.

## Dependencias de prompts y respuestas derivadas

`PromptContext` conserva el texto seleccionado junto con las capturas y recibos
de hechos efectivamente incluidos. Cada recibo contiene ID, revisión y plazo,
sin duplicar contenido. Se enlazan al intercambio actual y sobreviven en sus
fuentes, memoria corta, turnos, resúmenes, snapshots, cachés y eventos. Unir
fuentes del mismo UUID conserva ambas dependencias y el menor plazo; no permite
que una copia menos completa reemplace a la enriquecida.

Un hecho semántico sigue siendo consultable después de vencer su captura
literal, durante sus 30 días o mientras siga confirmado. El prompt no vuelve a
introducir por esa vía el texto original de una captura vencida. Una respuesta
cruda conserva como máximo 48 horas y el menor plazo de las entradas que usó.
Un hecho automático extraído de esa respuesta tampoco renueva el plazo de sus
padres semánticos. El catálogo guarda sus relaciones y el borrado de un padre
retira transitivamente los hijos y encola sus eliminaciones del índice en la
misma transacción. Los resultados tardíos no pueden recrearlos.

La proyección de autoridad para el loop contiene sólo recibos, no contenido ni
conexiones SQLite. Comprueba revisión, vencimiento y revocaciones sin disco ni
locks de SQLite. Los IDs afectados se bloquean antes de que el commit libere
el GIL; la autoridad positiva nueva se publica después del commit. Un commit
fallido vuelve a consultar el estado real tras rollback, sin suponer que no
llegó a disco. Una reconstrucción de índice o un borrado Chroma fallido no
restaura permiso para usar un hecho eliminado.

La cola conversacional adjunta tanto el historial/resumen usado como la
selección de memoria. Vigila las fuentes durante generación y entrega. Limpiar
o reemplazar el contexto invalida además su época, aunque las capturas sigan
vigentes. El reasoner vuelve a comprobar admisión en el punto de enviar el
prompt, incluso después de discovery. Cada fragmento se valida antes de
publicarse y los recibos terminales conservan las dependencias hasta el router
que registra el intercambio. Una cancelación por privacidad no detiene el
worker de la cola ni genera un aviso audible de error.

El modo legacy aplica la misma selección y vigilancia a la respuesta profunda,
al contexto del router rápido y a su reproducción. La operación nativa rápida
comprueba fuentes al admitir trabajo y al devolverlo; sigue bajo seguimiento si
su observador se cancela. El monitor conserva el cierre de la entrega incluso
con cancelaciones repetidas. Esto no reemplaza la coordinación pendiente de
los demás productores ni promete interrumpir por fuerza un motor síncrono.

Una propuesta permanente sigue partiendo de una afirmación directa del usuario.
La API no acepta convertir una derivación con recuerdos heredados en una
propuesta de origen directo; una nueva captura explícita puede proponerse y
requiere la revisión autenticada existente.

La actualización del catálogo agrega almacenamiento de dependencias bajo la
lease del único escritor. Los hechos automáticos del esquema previo, cuya
independencia no se puede verificar, conservan su contenido en cuarentena y su
vencimiento original; sus filas vectoriales dejan de autorizar búsquedas y se
encola su retirada. Los confirmados manualmente conservan su autoridad. Los
snapshots nuevos usan versión 3: no se inventa procedencia para los resúmenes de
versión 2, y su purga sigue el plazo original. Estas pruebas usan bases
temporales. Los 870 hechos de la colección legacy y los datos de producción no
se modificaron; la migración y restauración con datos reales en copia siguen
pendientes. Volver a un runtime anterior requiere también datos compatibles.

El procesador `RequestProcessor` antiguo de `priority_queue.py` no está cableado
en el grafo de runtime; conserva una API vieja de prompts y queda señalado para
la auditoría de rutas alternas antes de H6. La garantía probada aquí corresponde
a `MultiUserOrchestrator` y a ambos modos del `RequestRouter` activo.


La [validación de procedencia de prompts](../research/2026-10-26-h3-prompt-provenance-validation.json)
del commit `951bc4a` registra **4434 passed, 1 xfailed**, sin fallos, en 158,11 s.
Los 29 casos nuevos cubren ambos modos activos, revocación durante discovery,
lecturas tardías, historial y épocas de contexto, derivados semánticos, commit
incierto, reinicio, esquema anterior y lease, snapshots y cierre cancelado.
Pasan Ruff, Bandit medio/alto (30 bajos de la línea base), 18 tests Node, smoke
del dashboard y artefacto compilado con recibos y borrado transitivo. El heartbeat
de la suite obtuvo p95 de 0,19 ms y máximo de 1,02 ms frente a 25/75 ms. No se
publicó una release y H3 sigue abierto.


## Autoridad independiente de la memoria semántica

`main` construye `PrivacyServices` aunque `memory.enabled` sea falso. Conserva
`memory.privacy_db_path` como configuración del archivo persistente; el factory
rechaza un catálogo volátil. NLU, eventos, contextos, cola, ambos modos de voz y
captura ambiental reciben el mismo catálogo. Apagar la memoria semántica no elimina las
revocaciones. La API de revisión semántica sigue informando indisponibilidad
cuando esa función está apagada, y no se inicializan sus preferencias ni índice.

La autoridad dispone de dos threads de I/O propios y admite hasta ocho trabajos
retenidos. Una búsqueda semántica bloqueada en el executor general no ocupa ese
executor de privacidad. El test que satura el executor general conserva la
revocación; no implica que SQLite pueda saltarse un writer o disco bloqueados.
El manager opcional registra el cierre real de su trabajo nativo como usuario
del catálogo. Aunque venza la observación de su cierre, la autoridad espera ese
trabajo antes de cerrar SQLite, unir sus threads y liberar la lease. Un segundo
writer sigue rechazado durante esa espera. La inicialización cancelada también
conserva su cleanup registrado antes de empezar I/O.

La proyección del catálogo admite lecturas sólo mientras conserva autoridad.
Cerrar la autoridad invalida los recibos viejos. Eventos y snapshots distinguen
indisponibilidad de revocación: una consulta o purga falla cuando no se puede
consultar la autoridad, sin interpretar ese estado como un borrado realizado.
Los archivos de contexto existentes se conservan durante esa indisponibilidad.

`CaptureScopes` comparte la agrupación por origen/habitación entre privacidad
y memoria. Una revocación conserva su fecha en el catálogo. La captura anterior
a esa fecha no recibe un UUID de conversación nuevo; el límite se conserva al
rotar al intercambio posterior. Después de reiniciar, una captura sin
conversación que precede al arranque se rechaza: no se inventa un scope nuevo
para un buffer anterior. Las capturas posteriores siguen funcionando; las
fuentes ya identificadas conservan sus IDs y se validan contra el catálogo.
No se usan estos scopes como identidad de una persona ni como permisos.

El transcriber ambiental fija la procedencia antes de esperar al STT y comprueba
su vigencia al volver. Si perdió la fuente, descarta el texto antes del tagger.
El store configurado con autoridad exige esa conversación previa, impide
reinserciones revocadas y conserva el menor vencimiento recibido. Sus consultas
SQL filtran revocaciones; también se revalidan las filas después de esperar al
cursor y el clip después de leer el archivo. La purga registra los FLAC para
borrado durable antes de retirar sus filas; un unlink fallido se reintenta tras
reinicio. El debug ambiental ya no copia transcripciones ni nombres de hablante.

Este paso conecta las garantías de los componentes; todavía no expone un
borrado global en el dashboard. Faltan el inventario de participantes, el journal
común de operaciones y reintentos, la cancelación de lotes de destilación en
curso y los demás productores. Si el índice semántico está deshabilitado, sus
eliminaciones físicas siguen pendientes en el journal del índice; el futuro
coordinador debe resolverlas sin depender de habilitar la memoria semántica. Los datos de producción
y la colección legacy no se modificaron.


La [validación de autoridad independiente](../research/2026-10-26-h3-privacy-authority-validation.json)
del commit `74edec5` registra **4459 passed, 1 xfailed**, sin fallos, en 159,57 s.
Los 25 casos nuevos incluyen memoria deshabilitada, conversación y control local
vigentes en ambos modos, reinicio, pool general saturado, lease durante cierre
retenido, STT tardío, cursores revocados, unlink durable y estado desconocido.
La primera suite pasó en `3008701`; la revisión posterior reprodujo tres fallos
sobre snapshots con autoridad cerrada y capturas anteriores al arranque. Se
corrigieron antes del candidato final y los tres casos corren en su suite.
Pasan Ruff, Bandit medio/alto, 18 tests Node, smoke y artefacto compilado con
memoria opcional apagada. El heartbeat obtuvo p95 de 0,23 ms y máximo de 1,09 ms
frente a 25/75 ms. No cierra H3 ni constituye una publicación en producción.

## Avance: intención durable y participante del índice

El catálogo registra ahora la revocación, el scrub de hechos y una operación de
borrado en la misma transacción. Repetir el pedido conserva ID, actor y fecha
originales. Los tombstones anteriores reciben una operación pendiente al
arrancar, conservando sus metadatos originales. El estado distingue fuentes
revocadas de limpieza completa; la ausencia de un participante bloquea esa
última declaración. No se agregó todavía un endpoint global de borrado.

El manifiesto obligatorio incluye productores, capturas/clips, contextos,
índice semántico, eventos, diagnósticos, colectores, streams, sugerencias y voces.
Ese avance conectó sólo el driver del índice. Los demás conservaban recibos pendientes
hasta implementar la coordinación de sus productores y copias. El inventario
encontró texto sin política común en logs estructurados, métricas JSONL y
Logstash, el bus de eventos vivos y los JSON del colector de conversaciones.
El colector actual se construye en el bloque nocturno; no se presupone que esté
inyectado en el router. Estos hallazgos siguen abiertos.

El registro privado conserva todos los destinos y versiones de almacenamiento
conocidos. Agregar una ubicación reabre también operaciones anteriores; cambiar
la configuración no descarta destinos viejos. El factory semántico registra su
destino antes del primer acceso al índice, incluso si luego falla el arranque.
Los recibos de estado no contienen rutas, descriptores, cursores, tokens de claim
ni texto de excepciones. Esta lista todavía necesita reconciliarse con el
inventario de datos reales antes de la release y no descubre ubicaciones que
nunca estuvieron registradas.

Cada intento tiene un token exclusivo; un worker viejo no puede confirmar otro
intento. La recuperación de claims interrumpidos ocurre una sola vez, después
de adquirir la lease del runtime. Abrir otro catálogo o repetir initialize no
reinicia un claim vivo. El coordinador conserva hasta cuatro callbacks reales
y el handoff desde claim, aunque se cancele el observador. Si falla el guardado
del recibo, conserva el callback terminado y reintenta ese guardado sin duplicar
la limpieza. El resultado retenido se normaliza a un DTO sin texto de excepción,
traceback ni campos adicionales del driver; tampoco se conserva un resultado
inválido con contenido privado mientras falla la escritura del recibo.
Un timeout de cierre mantiene callbacks, executor y lease hasta
su salida real. Los fallos se registran con códigos fijos y backoff acotado.

El participante Chroma funciona con la memoria semántica apagada. Usa un worker
dedicado y acceso de limpieza sin embeddings ni creación de colecciones. Recorre
en lotes de 32 los IDs ya eliminados en el catálogo; el cursor se guarda tras
completar el lote. Comparte la barrera del writer activo, de modo que una escritura
que leyó el hecho antes de revocarlo termine antes de confirmar su eliminación.
Una colección o directorio ausente no se crea. La colección legacy, los hechos
vigentes y la cuarentena se conservan. La cola del índice actual se confirma por
revisión; los destinos históricos tienen recibos propios. Esto cubre la limpieza
del índice para las operaciones de conversación registradas: la reconciliación
independiente de todos los vencimientos y borrados individuales con recall
apagado sigue pendiente.

Las pruebas conectadas usan SQLite y Chroma temporales: atomicidad con rollback,
idempotencia, permisos, participantes y versiones ausentes, cambio de destino,
reinicio con cursor, fallos de limpieza/recibo, cancelación durante claim,
lease retenida durante trabajo nativo y upsert tardío. La aceptación del grafo
completo, la migración real y la conservación de la intención de borrado durante
restore continúan pendientes. No se modificó producción.

La [validación del journal y del índice](../research/2026-10-26-h3-erasure-journal-validation.json)
del commit `572aad5` registra **4484 passed, 1 xfailed**, sin fallos, en 161,58 s.
Pasan los 25 casos nuevos, Ruff, Bandit medio/alto, 18 tests de frontend, smoke
y artefacto compilado. El heartbeat mantiene los límites de 25/75 ms: p95 de
0,18 ms y máximo de 1,00 ms. El primer candidato pasó la suite; la revisión
posterior reprodujo la retención de una excepción original cuando fallaba su
recibo. El candidato final incluye su corrección y tres regresiones para
excepciones, resultados inválidos y campos extra. No cierra H3.

## Avance: participante de capturas y archivos tardíos

El runtime conecta también `CaptureEraser` al journal antes de construir el
ambient path. Registra la base y el directorio de audio aunque ambient o keep_audio
estén deshabilitados. Antes de arrancar productores, attach verifica la misma
autoridad y store del archiver y registra sus ubicaciones efectivas. Los destinos
anteriores siguen siendo obligatorios. Las rutas volátiles o URI no pueden
sustituir la persistencia de las capturas gestionadas por esta política.

Cada clip nuevo con autoridad usa `policy_v1/<capture_uuid>.flac`; dos bases con
el mismo rowid no comparten archivo. Antes del encoder, el store guarda su ruta,
conversación y vencimiento en `audio_write_intents`. Así se conserva el localizador
si el proceso cae entre escribir y adjuntar el clip. El adjunto aceptado retira
el intent; un adjunto cancelado o rechazado pasa por la cola durable de unlink.
Los archivos y la base nueva son privados (0600); el directorio final nuevo es
0700. No se renombraron ni modificaron clips reales existentes.

El archiver limita a ocho sus escrituras nativas retenidas. Valida la fuente
antes de reservar, después de reservar, al entrar al worker y al volver del
encoder. La reserva devuelve el menor vencimiento efectivo del store y de la
fuente: un encoder lento no usa el plazo mayor de 48 h cuando su fila vence
antes. El transcriber la comprueba nuevamente después de esperar al wake
textual. Una captura revocada no inicia otro archivo ni publica un resultado
tardío. El encoder que ya entró permanece registrado hasta su salida real,
incluso si se cancela repetidamente al llamador.

El participante espera los encoders de la conversación antes de retirar sus
filas, scope y localizadores. La purga TTL conserva intents y pospone unlink
mientras su encoder siga activo; por eso no pierde la ruta si corre antes que
el borrado. Al retirar filas, el store registra primero los archivos pendientes
en la misma transacción. Un unlink fallido deja la operación pendiente con su
contador, incluso tras reiniciar. Los clips y texto de otra conversación se
conservan. La lease de privacidad espera también los encoders vinculados cuando
se agota el presupuesto de observación del cierre.

La limpieza abre sólo bases existentes. Una base ausente no se recrea; si quedan
FLACs sin base que permita atribuirlos, el destino sigue pendiente. Una base
corrupta o un directorio ilegible tampoco equivalen a ausencia. El recorrido de
audio no usa rglob como prueba de vacío, porque Python 3.13 puede ocultar errores
de recorrido. Las rutas de archivos se restringen a las raíces registradas para
esa base: incluye un keep_audio.dir anterior, pero no una ruta arbitraria en
una fila. Un localizador fuera de esas raíces queda pendiente, sin eliminar
archivos ajenos. No se atribuyen conversaciones inventadas a clips legacy.

Este paso conecta dos de los diez participantes, capturas e índice. No drena
todavía STT, lectores de audio, buffers del tap ni lotes de destilación; los
participantes de productores, voces y streams deben cubrir esas copias. Tampoco
agrega un endpoint global ni declara completo el borrado de conversación. Esa
versión aún no mantenía la purga de 48 h sobre todos los destinos registrados con
ambient apagado y raíces anteriores sin un pedido de borrado; dependía del
transcriber activo. El siguiente avance resuelve esa dependencia. La migración y
restauración de datos reales y el resto de H3 siguen pendientes.

La [validación del participante de capturas](../research/2026-10-26-h3-capture-erasure-validation.json)
del commit `63495dc` registra **4507 passed, 1 xfailed**, sin fallos, en 162,85 s.
Incluye 23 casos nuevos, SQLite y FLAC reales temporales, 309 pruebas dirigidas,
Ruff, Bandit medio/alto, 18 tests de frontend, smoke y artefacto compilado. El
heartbeat mantiene p95 de 0,39 ms y máximo de 1,24 ms frente a 25/75 ms. El primer
intento completo falló porque el hook de una prueba de cancelación conservaba
la firma anterior de la cola; se adaptó conservando sus comprobaciones de
atomicidad. La revisión adicional reprodujo el encoder que usaba un plazo mayor
que su fila; el candidato final incluye su corrección y regresión. La evidencia
conserva ambos resultados y el aviso inicial de Bandit, resuelto con una consulta
literal parametrizada y el guard de vigencia. Esto no cierra H3.

## Avance: purga independiente del ambient path

`CaptureRetention` se construye y arranca con el participante de capturas, antes
de los productores. Recorre las bases y raíces históricas del registro privado
aunque ambient, keep_audio y memoria semántica estén apagados. No carga STT,
SpeakerID, Chroma ni embeddings para hacerlo. Las ubicaciones nuevas se registran
antes de admitir archivos y despiertan su mantenimiento; los destinos previos
siguen incluidos después de cambiar configuración o reiniciar.

El planificador mantiene hasta dos bases en trabajo independiente y un executor
de disco propio de dos threads. Una búsqueda semántica o un encoder en el pool
general no ocupa esos threads. Una base corrupta o un unlink bloqueado no
impiden que otro destino independiente se limpie. El trabajo real de cada barrido
sigue registrado si se cancela su observador. El estado de diagnóstico sólo
contiene IDs opacos, contadores y códigos fijos; no contiene rutas ni texto de
errores privados. Un destino o versión indeterminado queda visible y no autoriza
eliminaciones que no se puedan atribuir.

La fecha persistida de cada fila/intent limita la retención. El mantenimiento
respeta también un límite configurado menor que 48 h, sin renovar fechas al leer
o reabrir. Las capturas nuevas y los archivos pendientes despiertan el scheduler;
se consulta el vencimiento más próximo y se acotan reintentos a 60 s. El recibo
guarda cuándo se calculó el próximo plazo: recogerlo tarde no le suma otra espera
al vencimiento original. No se interpreta una lectura filtrada como prueba de
que ya terminó la limpieza física.

El barrido retira texto y scopes vencidos y conserva los intents de encoders
activos hasta su salida. Las rutas pasan a la cola durable antes del unlink.
Para evitar colisiones de datos anteriores, el guard de archivos comprueba sólo
metadatos en todas las bases registradas que comparten esa raíz. Una referencia
vigente protege el archivo; una base inaccesible o de esquema desconocido también
lo conserva, con la limpieza pendiente. Un rowid vigente de otra base protege
el archivo numérico legacy aunque todavía no tenga puntero adjunto.

Los huérfanos numéricos legacy siguen la guarda previa de mtime y se encolan en
lotes de hasta 256, sin borrar rutas ya pendientes de nuevo ni tocar archivos
protegidos. El recorrido usa scandir y propaga errores, en vez de interpretar
un directorio ilegible como vacío. Archivos sin una base atribuible y protocolos
de archivo ajenos al conocido requieren inventario/revisión; no se les inventan
fuentes. Los clips nuevos con UUID disponen de su intent durable desde antes
del encoder. Todavía falta contrastar las ubicaciones reales y las copias legacy
en el ensayo de migración.

El store activo comparte guard, raíces históricas y ownership del disco con el
mantenimiento. La apertura/migración del schema se serializa con BEGIN IMMEDIATE
entre conexiones y hace rollback si falla. Esto evita que la conexión del
transcriber y el barrido agreguen simultáneamente una columna. El archiver ahora
retiene toda la llamada, incluida reserva y descarte, además del encoder: el
cierre detiene admisión y espera esas llamadas antes de cerrar disco y liberar
la lease. El presupuesto de observación y la tarea de cierre real siguen siendo
cosas distintas.

Este avance no conecta los otros ocho participantes del borrado global ni drena
STT, lectores, tap y destilación. La reconciliación de todos los vencimientos y
borrados individuales del índice con recall apagado sigue pendiente, al igual
que pausa/mute, voces, copias de diagnóstico y migración real. No cierra H3 y no
modificó producción.

La [validación del mantenimiento independiente](../research/2026-10-26-h3-capture-maintenance-validation.json)
del commit `22ebc58` registra **4527 passed, 1 xfailed**, sin fallos, en 163,30 s.
Los 20 casos nuevos cubren funciones apagadas, plazos anteriores, reinicio,
ubicaciones históricas, referencias compartidas, fallos de disco, pool general
saturado, cierre retenido y apertura concurrente del schema. Pasan 329 pruebas
dirigidas, Ruff, Bandit medio/alto, 18 tests de frontend, smoke y artefacto
compilado. El heartbeat mantiene p95 de 0,40 ms y máximo de 1,05 ms frente a
25/75 ms. La evidencia conserva las primeras carreras y la corrección de la
observación del cierre sin ampliar sus presupuestos. No cierra H3.

## Avance: limpieza de hechos sin recall

El índice de hechos tiene ahora recibos durables por destino, ID y revisión en
`policy_index_deletions`. Borrar un hecho o vencer su plazo registra todas las
copias conocidas dentro de la misma transacción que retira su contenido del
catálogo. Esto incluye recuerdos confirmados borrados manualmente, propuestas
rechazadas/vencidas y hechos automáticos de hasta 30 días. El mantenimiento se
construye siempre con el driver del índice, sin habilitar la memoria semántica.

Registrar otro destino incorpora también hechos ya eliminados o en cuarentena,
aunque el índice activo haya confirmado su cola anterior. Los recibos conservan
revisión y token de intento; no hay expiración automática de un claim cuyo
trabajo nativo siga vivo. Un recibo con versión, descriptor o token incorrectos
no confirma otro trabajo. El código vuelve a comprobar la autoridad antes de
abrir el índice. Un estado de hecho desconocido deja la limpieza pendiente.

El arranque con lease reabre los recibos para comprobar el estado físico de cada
índice: restaurar una proyección anterior no puede esconderse detrás de una
confirmación vieja. También reabre el paso semántico de operaciones de borrado
ya completadas. Repetir initialize o abrir un catálogo auxiliar no reinicia
claims vivos. Dentro de un mismo arranque, el borrado de conversación y el
mantenimiento comparten recibos para no duplicar eliminaciones ya confirmadas.
El recibo de conversación sigue pendiente si quedan trabajos de índice que un
cursor sobre filas deleted no puede demostrar resueltos.

Los lotes tienen hasta 32 IDs, dos destinos en trabajo y dos threads dedicados,
con ocho operaciones nativas retenidas como máximo. El destino activo usa la
barrera del writer real; los destinos históricos independientes tienen sus
propias barreras. Un upsert que leyó antes del borrado termina antes de publicar
el recibo físico. El factory abre la ubicación canónica que registró antes de
construir el cliente, y el limpiador conserva ese descriptor del writer incluso
si cambia después el alias configurado.

La limpieza abre sólo colecciones existentes de la política, sin embeddings ni
creación de colecciones/directorios. Las ausencias comprobadas se concilian; una
versión no soportada o un fallo queda pendiente con un código fijo. Si el trabajo
nativo terminó y falla el recibo, se conserva su resultado sin excepción privada
y se reintenta sólo el guardado. Los fallos físicos usan backoff hasta 60 s.
Una reserva de claim o un trabajo nativo sobreviven a la cancelación del
observador; su cierre real precede a liberar la lease.

Las mutaciones despiertan el mantenimiento, incluso ante un plazo nuevo anterior
al que estaba esperando. La cadencia no depende del pool general ni de arrancar
recall. El planificador consulta pendientes indexados, sin recorrer repetidamente
el historial de recibos completados. `pending_index_cleanup` informa las copias
todavía pendientes; la cola anterior del índice activo por sí sola no prueba
que los destinos históricos terminaron.

Los hechos vigentes y confirmados se conservan. El contenido del catálogo que
ya estaba en cuarentena se mantiene fuera de inferencia, mientras se retira su
proyección de la colección de política. No se toca la colección legacy original
de los 870 hechos. No se reconstruyen ni publican hechos vivos en índices
históricos con recall apagado. La reconstrucción de un índice vivo nuevo/vacío
y el inventario de datos sin autoridad siguen siendo parte de la aceptación.

Esto completa la vía de limpieza de vencimientos y borrados individuales sobre
los destinos registrados. No demuestra una restauración del catálogo antiguo:
el ensayo compatible debe conservar la autoridad/tombstones recientes antes
de admitir datos o prompts. El resto de participantes, productores, copias,
interfaz autenticada del borrado global, voces y migración real siguen abiertos.
No cierra H3 ni constituye un cambio en producción.

La [validación del mantenimiento del índice](../research/2026-10-26-h3-index-maintenance-validation.json)
del commit `5faa945` registra **4557 passed, 1 xfailed**, sin fallos, en 175,95 s.
Incluye 28 casos nuevos del índice, dos regresiones del arranque del logger,
119 pruebas dirigidas, Ruff, Bandit sin hallazgos medios/altos, 18 tests de
frontend, smoke y artefacto compilado. El ensayo compilado comprueba borrado
individual y vencimiento sin recall ni tombstone de conversación. El heartbeat
de voz obtiene p95 de 0,21 ms y máximo de 1,07 ms frente a 25/75 ms.

La evidencia conserva el primer fallo de la suite: el fixture admitía eventos
antes de inicializar el logger, mientras el factory real inicializa antes de
aplicar el presupuesto de cada escritura. La corrección sigue ese orden y prueba
tanto el arranque lento como la continuidad del trabajo nativo tras un timeout
en frío. No se amplían los 100 ms. La observación adicional de GC, sin desactivarlo
ni cambiar sus umbrales, registra 42 pausas de al menos 10 ms, 41 superiores a
75 ms y un máximo de 495,68 ms fuera de los casos de latencia de voz. Las etiquetas
de test no establecen el origen de los objetos retenidos ni una causa de runtime.
La aceptación integrada de GC/latencia en H4/H6 sigue abierta; pasar la suite no
la sustituye.

## Avance: participante de eventos

El grafo de `main` registra ahora el participante `events` además de capturas e
índice. Se construye aunque analytics esté apagado, registra la ubicación actual
y conserva las históricas del journal. El writer registra la ubicación canónica
antes de abrir SQLite y sigue usando esa ubicación si cambia un alias posterior.
Un destino añadido después de una solicitud recibe su propio paso pendiente.

El adaptador abre sólo bases existentes en modo `rw`, sin inicializar schemas ni
crear archivos. Recorre hasta 128 IDs y fuentes por lote; no carga frases,
nombres ni contextos. Retira eventos que dependen de la conversación revocada,
incluidas fuentes mixtas y recibos de hechos derivados ya retirados. Conserva
eventos ajenos y métricas sin texto. La tabla legacy sigue intacta y fuera de
las consultas: completar este destino demuestra sólo la limpieza de
`events_policy_v1`, sin atribuir conversaciones a filas legacy.

Fuentes malformadas, texto sin atribución, IDs no reconocidos, bases corruptas
y versiones desconocidas no producen una confirmación falsa. Un lote con
procedencia desconocida revierte completo. Los recibos públicos conservan
códigos fijos, sin rutas ni contenido. El arranque con lease reabre los pasos de
eventos desde el principio para comprobar posibles filas restauradas detrás de
un cursor o recibo anterior; repetir initialize no reinicia un claim vivo.

La admisión existente del writer conserva el commit bajo la barrera de fuentes:
o termina antes del tombstone y el adaptador lo encuentra, o se rechaza. Un
provider tardío que sobrevive al timeout no puede reponer texto borrado. Writer y
adaptador tienen cada uno dos threads propios y ocho operaciones retenidas como
máximo; el provider ya no ocupa el pool de la autoridad. El factory registra el
writer como prestatario de la lease antes de inicializarlo. El cierre cierra
admisión y conserva mantenimiento, llamadas nativas y pools hasta su salida real.

Quedan siete participantes sin conectar y las copias derivadas de eventos en
sugerencias, colectores y streams. Este primer avance no independizaba la purga
de TTL del logger activo; ese trabajo se describe en el avance siguiente.
El borrado de conversación no autoriza una promesa global de borrado. No cierra H3.

La [validación del participante de eventos](../research/2026-10-26-h3-event-erasure-validation.json)
del commit `4c297b3` registra **4582 passed, 1 xfailed**, sin fallos, en 169,95 s.
Incluye 25 casos conectados nuevos, 132 pruebas dirigidas, Ruff, Bandit sin
hallazgos medios/altos, 18 tests de frontend, smoke y artefacto compilado. El
ensayo compilado comprueba tanto el recibo terminado como cero filas físicas
del evento borrado. El heartbeat obtiene p95 de 0,22 ms y máximo de 1,07 ms;
la cancelación de control alcanza un máximo de 0,07 ms y el cleanup 11,13 ms,
sin ampliar límites. La aceptación de GC/latencia integrada sigue pendiente:
esta suite sin instrumentación de GC no vuelve a medir ni resuelve las pausas
globales observadas en la validación anterior.

## Avance: retención de eventos sin analytics

El adaptador de eventos construye ahora un mantenimiento independiente del
logger, del feature analytics y de recall. Recorre las bases registradas,
incluidas las históricas, sin crear archivos ni schemas ni modificar tablas
legacy. Cada transacción procesa hasta 128 IDs, fechas y fuentes: borra eventos
con metadatos vencidos o dependencias revocadas y reemplaza frases, nombres y
contextos vencidos por las métricas numéricas ya filtradas. No carga texto en
Python. Las fechas persistidas no se renuevan; un límite de metadatos menor se
guarda antes de vencer para que ampliarlo después tampoco lo extienda.

Dos lotes y dos threads propios limitan el trabajo nativo. El planificador cede
entre lotes para que una base grande no monopolice los destinos pendientes.
Un SQLite bloqueado no conserva la barrera de la autoridad ni frena otro destino
independiente. Las escrituras del factory despiertan el mantenimiento ante fechas
anteriores; el poll del registro y las revocaciones se limita a 30 s. El factory
conectado usa este mantenimiento en lugar de duplicarlo con el barrido completo
del logger. El logger standalone conserva su mecanismo existente.

Las filas y sus fechas son la intención durable de purga: tras fallo o reinicio
se vuelven a examinar. Fuentes o fechas desconocidas revierten el lote y quedan
pendientes con un código fijo; las versiones desconocidas no se confirman.
Los resultados retenidos contienen sólo contadores, cursores y fechas. La espera
cancelada no cancela el lote, y el cierre conserva su pool y la lease hasta su
salida real. Los deadlines se anclan al momento de calcularlos para no extender
la espera cuando el loop observa tarde un resultado.

La revisión detectó además que una revocación provisional previa al commit no
sirve como prueba para destruir una copia física. Los lectores siguen bloqueados
durante ese commit, pero la limpieza irreversible usa sólo las revocaciones y
recibos publicados tras un commit confirmado. Esto cubre al nuevo mantenimiento,
al participante del journal y al borrado del logger standalone. Un caso mantiene
el commit pendiente, ejecuta las dos vías físicas y fuerza rollback: el evento
válido se conserva y vuelve a estar autorizado.

La revisión de lectura reprodujo dos huecos en filas sintéticas: un objeto JSON
vacío se interpretaba como una lista vacía de fuentes, y una fecha raw guardada
posterior al vencimiento de la fuente podía autorizar texto hasta la purga.
La vista SQL valida ahora la forma de las fuentes y las fechas de metadatos;
el acceso a frases, nombres y contexto exige también que las fuentes reales
sigan dentro de su plazo. Las métricas válidas pueden conservarse durante su
vigencia propia. Los casos comprueban todas las proyecciones analíticas y que
la exclusión no depende de borrar físicamente la fila.

No completa los otros siete participantes ni las copias derivadas, la migración,
voces o el resto de H3. Las ubicaciones nunca registradas y los datos sin
procedencia todavía requieren el inventario privado y el ensayo de migración.

La [validación de retención de eventos](../research/2026-10-26-h3-event-retention-validation.json)
del commit `d6dd9bc` registra **4607 passed, 1 xfailed**, sin fallos, en 176,39 s.
Incluye 25 casos nuevos, 157 pruebas dirigidas, Ruff, Bandit sin hallazgos
medios/altos, 18 tests de frontend, smoke y artefacto compilado. El ensayo
compilado cierra el logger antes de avanzar a las 48 h y comprueba la purga de
texto/contexto por el mantenimiento independiente, sin tombstone. La evidencia
conserva la primera suite sin fallos y los huecos de lectura reproducidos que
motivaron cinco casos adicionales y la corrección del candidato final.
El heartbeat obtiene p95 de 0,21 ms y máximo de 1,73 ms; la cancelación de control
alcanza un máximo de 0,15 ms y el cleanup 11,09 ms con los límites intactos.
No se vuelve a medir ni se da por resuelta la pausa global de GC previa:
la aceptación integrada de runtime/GC en H4/H6 sigue abierta.

## Avance: participante de contextos y snapshots

El grafo de `main` conecta ahora el participante `contexts`, con destinos para
los buffers en memoria y para cada directorio físico de `policy_v2` y
`policy_v3`. Se registran las ubicaciones aunque estén apagados orchestrator,
persistence y recall; las anteriores permanecen en el journal. El persister
registra su ubicación canónica antes del primer acceso y la conserva aunque
cambie luego un alias configurado. El arranque con lease reabre recibos y
cursores para volver a comprobar archivos restaurados; initialize repetido no
recupera claims de un propietario vivo.

El adaptador elimina snapshots con fuentes de la conversación o dependencias
revocadas, conserva archivos ajenos y no toca los JSON legacy del directorio
original. Cada lote recorre hasta 32 entradas, con cursor hash y hasta 1 MiB por
snapshot. No sigue symlinks de archivos. Procedencia desconocida, corrupción,
protocolos o tamaños no soportados quedan pendientes. Los borrados de archivo
ya realizados son idempotentes si otro archivo del lote falla; el recibo no se
confirma completo. Sólo los drafts reconocidos del protocolo se retiran como
restos no recuperables de escrituras interrumpidas.

Los persisters y el limpiador comparten la barrera del directorio físico. Esto
impide que otro persister borre un temporal aún en escritura. Un replace ya
admitido termina antes del tombstone y su archivo se encuentra después; un
payload que llega tarde al guard se rechaza. La purga existente del persister
también usa sólo revocaciones confirmadas para no destruir archivos ante un
commit provisional que luego falla.

En memoria se retiran turnos, resúmenes y entidades preservadas de los gestores
registrados, el buffer corto y las copias inactivas de consultas/classifier/cola
del router. Las admisiones tardías se vuelven a validar bajo sus barreras. Se
cancelan las compactaciones afectadas y se espera su salida, incluidas las
iniciadas por el cleanup. También se retienen hidrataciones y guardados nativos
ya admitidos, aunque sus observadores se hayan cancelado. Un resumen tardío no
puede publicarse sobre una revisión revocada. Los buffers sin procedencia y las
confirmaciones pendientes sin fuentes impiden confirmar el destino runtime.

El gestor usa dos threads propios y ocho operaciones I/O retenidas como máximo;
el adaptador de archivos tiene otros dos threads y ocho operaciones. Ambos
mantienen la lease hasta su cierre real. La espera del thread de cleanup es
interrumpible; el cierre no abandona llamadas delegadas. El registro de
prestatarios evita duplicar el mismo propietario. La limpieza de usuarios cede
entre grupos de 32 para no sostener el loop durante todo el recorrido.

Quedan seis participantes sin conectar. Este avance no completa los productores
activos de inferencia/audio, los otros consumidores, las copias derivadas ni la
interfaz de borrado amplio. Ese avance todavía no independizaba el TTL de los
features; el avance siguiente incorpora esa purga sobre ubicaciones conocidas.
El inventario/migración privada, voces y aceptación integrada de H6 siguen abiertos.

La [validación del participante de contextos](../research/2026-10-26-h3-context-erasure-validation.json)
del commit `5aec4fb` registra **4630 passed, 1 xfailed**, sin fallos, en 178,16 s.
Incluye 23 casos nuevos, 462 pruebas dirigidas, Ruff, Bandit sin hallazgos
medios/altos, 18 tests de frontend, smoke y artefacto compilado. La evidencia
conserva los primeros fallos del helper de quiescencia y de imports de tests;
la corrección observa también los destinos todavía debidos, sin cambiar
capacidad, backoff ni límites. El ensayo compilado comprueba borrado físico de
snapshot y de buffers incluso con los features apagados. El heartbeat obtiene
p95 de 0,20 ms y máximo de 1,03 ms; la cancelación de control alcanza 0,16 ms
y el cleanup 11,20 ms. La aceptación integrada de runtime/GC sigue abierta.

## Avance: vencimiento independiente de contextos

El mantenimiento de contextos se construye siempre con el adaptador del journal,
sin habilitar orchestrator, persistence ni recall. Recorre los directorios de
política actuales e históricos y retira snapshots con fuentes vencidas o
revocadas, sin inventar fechas a partir de lecturas, reinicios o mtime. Los
archivos legacy originales siguen intactos. Cada lote procesa hasta 32 entradas
con el límite existente de 1 MiB por archivo y cede a otros directorios entre
lotes. Los temporales reconocidos se retiran sólo bajo la barrera compartida
con los writers.

Los archivos sin procedencia, corruptos, de protocolo desconocido o con symlinks
se conservan pendientes y no impiden llegar a otros archivos atribuibles del
directorio. Los contadores se acumulan hasta terminar el recorrido; las fechas
conocidas conservan el deadline original. Los resultados retenidos no contienen
texto ni excepciones privadas. Los fallos del directorio usan backoff acotado.

La actualización del registro es una tarea retenida separada. Una escritura que
mantiene la barrera del catálogo no detiene el barrido de otros directorios ya
conocidos. El factory carga esos destinos antes de admitir writers opcionales.
Las escrituras despiertan el mantenimiento y las actualizaciones del registro
se observan sin renovar los plazos calculados anteriormente.

También se limpian los turnos, resúmenes, entidades preservadas y memoria corta
que siguen en objetos inactivos, sin depender de otra consulta. El recorrido
cede entre grupos de 32 usuarios. Una compactación vencida se cancela una vez
y sigue figurando como operación pendiente hasta su salida; no detiene la purga
de otros buffers. El mantenimiento de archivos usa dos threads propios y ocho
operaciones retenidas; cerrar o cancelar la espera conserva sus lotes, registro
y recursos reales hasta liberar la lease.

La lectura y escritura de snapshots exigen ahora fuentes incluso cuando el
resumen está vacío, para no hidratar nombres u otros metadatos sin plazo. Los
originales sin fuentes no se borran por esa exclusión. La validación de IDs usa
el mismo protocolo de nombre completo del limpiador. La revisión de pruebas
de saturación añadió una comprobación explícita de que el pool general sigue
ocupado hasta terminar la operación; el fixture suspende sólo el observador
de TTL y deja el cierre de su pool para después de liberar ese bloqueo.

Este avance cubre los contextos de política atribuibles y los buffers registrados.
No da procedencia a los archivos desconocidos, no sustituye la migración privada
ni completa los seis participantes restantes, la revisión de voces o H3/H6.

La [validación de retención de contextos](../research/2026-10-26-h3-context-retention-validation.json)
del commit `7606e42` registra **4652 passed, 1 xfailed**, sin fallos, en 183,55 s.
Incluye 22 casos nuevos, 508 pruebas dirigidas, Ruff, Bandit sin hallazgos
medios/altos, 18 tests de frontend, smoke y artefacto compilado. El ensayo cierra
el gestor antes de avanzar a las 48 h y verifica el archivo y los buffers
retirados sin consultas ni tombstone. La evidencia incluye el endurecimiento de
los casos de saturación y el fallo inicial de un import de test, corregido sin
ampliar límites. El heartbeat obtiene p95 de 0,22 ms y máximo de 3,07 ms;
la cancelación de control alcanza 0,09 ms y el cleanup 11,13 ms. La suite sin
instrumentación de GC no vuelve a medir ni resuelve sus pausas previas: la
aceptación integrada de runtime/GC y hardware sigue abierta.

## Avance: procedencia verificable de patrones de actividad

El analizador consume ahora una sola observación SQL acotada de metadatos,
sin cargar frases, nombres ni contexto libre. Cada patrón conserva recibos de
los eventos utilizados: destino opaco, ID de fila, huella de metadatos y fuentes,
y vencimiento original. La resolución consulta las filas reales y la autoridad
vigente; rechaza cambios, filas ausentes, revocaciones y recibos de otro destino.
La limpieza del texto a las 48 h no invalida los metadatos aún vigentes. Los
recibos conservan también el menor vencimiento de sus dependencias semánticas;
leer, recalcular o reiniciar con otra retención no renueva las fechas anteriores.
Estos recibos son referencias para revalidación, no permisos de publicación.

Las proporciones usan una ventana común y completa. Una secuencia cuenta cada
disparo una sola vez, aunque la acción siguiente se repita; su denominador
incluye disparos sin coincidencia dentro de la misma ventana temporal. Los
patrones semanales conservan ambos grupos usados en la comparación. El análisis
se abstiene si supera 10.000 eventos, y la parte de secuencias si supera 100.000
comparaciones; no transforma una muestra truncada en una confianza aparente.
Los contadores de diagnóstico no se convierten en automatizaciones domésticas.
Antes de devolver patrones se revalidan sus recibos, también si se borró una
fuente o venció mientras se calculaban.

El router ejecuta generación y respuestas de sugerencias fuera del loop,
serializadas dentro de su instancia y bajo el ownership nativo del logger real.
La cancelación del observador no libera ese trabajo ni permite que cierre el
logger antes de terminar. El canal de aceptación devuelve el mensaje real del
motor, sin anteponer una afirmación de automatización creada a una propuesta.

Queda integrar estos recibos en la política de carga, persistencia, presentación,
respuesta, vencimiento independiente y borrado durable de sugerencias, incluidos
los destinos históricos y los resultados tardíos. El JSON legacy y su lector
no se migran en este avance; no se consideran evidencia verificada. Los seis
participantes restantes, la revisión de voces, la migración privada y la
aceptación integrada de runtime/GC y hardware siguen abiertos. No se modifica
producción ni se habilita un borrado global.

La [validación de recibos de actividad](../research/2026-10-26-h3-pattern-evidence-validation.json)
del commit `568d1fd` registra **4682 passed, 1 xfailed**, sin fallos, en 192,61 s.
Incluye 30 casos nuevos, 142 pruebas dirigidas, Ruff en el alcance de CI, Bandit
sin hallazgos medios/altos, 18 tests de frontend y smoke de dashboard. El artefacto
compilado comprueba los recibos fuera del checkout en 6,48 s. El heartbeat tiene
p95 de 0,28 ms y máximo de 1,07 ms; la cancelación de control llega a 0,09 ms y el
cleanup a 11,23 ms, con límites intactos. La evidencia conserva el error inicial
del fixture que intentaba confirmar un hecho automático, corregido para probar
su borrado/revisión real y su vencimiento. También registra el reemplazo del SQL
dinámico por parámetros y los 50 hallazgos previos de scripts legacy ajenos al
alcance de lint de CI. No se vuelve a medir ni se resuelve la aceptación de GC.

## Avance: propuestas con almacenamiento de política

El motor activo deja de cargar o sobrescribir el JSON legacy. Las propuestas
nuevas usan una tabla SQLite separada junto a la ruta configurada, con archivo
privado y destino físico registrado antes de abrirlo. Cada lectura resuelve
los recibos contra los eventos reales, comprueba fechas y fuentes, y reconstruye
el patrón antes de proyectar el mensaje. Las filas con protocolo desconocido se
conservan fuera de las respuestas automáticas. No se mantiene un diccionario
privado que pueda reponer una fila retirada.

La descripción, el conteo y la consistencia se recalculan desde metadatos de
actividad: un `Pattern` recibido con texto o estadísticas alteradas no se toma
como fuente. El prompt del modelo opcional recibe esa proyección; el texto bruto,
los nombres y el contexto libre de los eventos no entran en él. La salida tiene
un límite de 4096 caracteres y los errores conservan sólo el tipo, con fallback
a la plantilla. El YAML se reconstruye desde la selección estructurada, sin
hidratar un YAML antiguo. La exportación de una hora cero conserva medianoche.

El plazo de cada propuesta es el menor de los eventos, sus dependencias
semánticas y 30 días desde la última observación real incluida, o el límite
más corto configurado para hechos automáticos. La fecha de generación se toma
antes de llamar al modelo. Volver a leer o calcular los mismos eventos, esperar
una generación, reiniciar, posponer o aceptar la propuesta no renueva ese plazo.
La vigencia de los metadatos permite patrones de varios días después de retirar
el texto a las 48 h; no autoriza a prolongar el contenido bruto.

Para confirmar una escritura, el motor toma primero la barrera de autoridad,
luego el bloqueo de escritura del registro de eventos, y finalmente el de
propuestas. Resuelve de nuevo los recibos y mantiene ambas barreras hasta el
commit. Una revocación queda después de un commit previo o rechaza una salida
posterior; una mutación de los eventos también espera ese commit. La generación
del modelo permanece fuera de las barreras para no bloquear el marker. La
resolución termina revalidando las fuentes de todos los lotes, no sólo las
últimas filas. Un marker pendiente que después falla oculta la propuesta sin
borrarla ni cambiar sus fechas; el rollback restaura su elegibilidad original.

La revisión encontró y reprodujo una ventana al reiniciar con menor retención
de eventos: antes del primer barrido aún se podían leer propuestas basadas en
plazos anteriores. La inicialización nativa del logger persiste ahora las
reducciones de las filas de política reconocidas antes de admitir consumidores.
Volver a aumentar la configuración no restaura el plazo viejo. Las filas de
fuentes desconocidas se conservan fuera de inferencia y no impiden acortar las
reconocidas. Es una actualización SQL de inicialización; el coste de arranque
con historiales grandes sigue dentro de la aceptación integrada de H4/H6.

La voz descarta la propuesta pendiente al vencer o revocarse y cancela su entrega
si pierde una dependencia. Los errores de lectura no convierten en fallida una
acción doméstica ya terminada; los de escritura no se presentan como aceptación
exitosa. Aceptar registra una propuesta y entrega su borrador; el stub anterior
ya no la etiqueta como automatización implementada. La ejecución efectiva de
rutinas sigue dentro de H5 y debe usar el servicio de control compartido.

La CLI usa la configuración existente, la misma autoridad con lease exclusiva y
el scope nativo del logger. No abre otro escritor si el runtime conserva la
lease, y cierra los recursos al terminar. La consulta de duplicados usa el índice
de la selección completa. El protocolo acota 1000 propuestas aún vigentes,
10.000 recibos/fuentes y 4 MiB por campo de metadatos; rechaza ubicaciones que
compartan archivo con eventos o con la autoridad, evitando bloqueos circulares.

La exclusión por TTL y revocación al leer y las barreras de escritura ya están
activas. Falta conectar el mantenimiento físico independiente de features y el
adaptador durable de sugerencias, incluidos los destinos históricos y los
buffers/productores restantes. Registrar destinos no equivale a completar ese
participante: siguen siendo seis los pendientes. No se habilita un borrado global,
no se migra el JSON original y no se modifica producción. La aceptación de GC,
hardware, voces, migración privada y el resto de H0–H6 continúa abierta.

La [validación del almacén de propuestas](../research/2026-10-26-h3-suggestion-store-validation.json)
del commit `3242ef5` registra **4721 passed, 1 xfailed**, sin fallos, en 192,57 s.
Incluye 39 casos nuevos, 280 pruebas dirigidas, Ruff con la CLI modificada,
Bandit sin hallazgos medios/altos, 18 tests de frontend y smoke del dashboard.
El artefacto compilado verifica creación, recarga, aceptación como borrador y
exclusión de propuestas aceptadas después del marker, fuera del checkout, en
6,57 s. El heartbeat tiene p95 de 0,50 ms y máximo de 1,08 ms; la cancelación de
control llega a 0,19 ms y el cleanup a 11,37 ms, con los límites intactos.

La evidencia conserva la primera suite completa verde del commit `b5069e6`,
la reproducción posterior del hueco de retención antes del barrido y su
corrección. También registra el error inicial del fixture de locators, el SQL
convertido a consultas literales y el refuerzo del smoke compilado para que
aceptar la única propuesta no volviera trivial el chequeo de pendientes tras
borrarla. La suite final incluye ese refuerzo y las pruebas de 90 → 1 → 90 días.
No completa el mantenimiento físico de propuestas, su participante durable ni
la aceptación integrada de GC y hardware.

## Avance: participante durable de sugerencias

El journal conecta ahora los sidecars actuales e históricos de propuestas y un
destino de runtime. Se registran antes de admitir los productores opcionales,
también con analytics apagado. La limpieza abre únicamente archivos de política
existentes y valida su esquema; no crea datos, sigue redirecciones por symlink
ni lee mensajes, descripciones o YAML. Recorre hasta 32 filas por transacción,
materializando un solo array de metadatos cada vez. Retira propuestas de cualquier
estado que incluyan la conversación revocada, fuentes mixtas o dependencias
semánticas ya retiradas; conserva las ajenas y el JSON legacy.

La procedencia y las fechas desconocidas quedan pendientes sin impedir que se
procesen las filas conocidas posteriores. También se conserva en cuarentena un
ID de fila que no podría producir el writer, sin hidratarlo ni saltarlo en el
recibo final. El cursor versionado conserva el avance y el conteo de desconocidos.
Los recibos incompletos no certifican el borrado de esa cuarentena. Un reinicio
con lease vuelve a comprobar los destinos de sugerencias desde el principio,
incluso los que tenían recibo completo, para detectar copias restauradas detrás
del cursor. Una versión no soportada o un fallo nativo quedan visibles con
códigos fijos, sin publicar rutas ni contenido privado.

La prueba de runtime toma un conjunto de operaciones del logger ya admitidas y
flujos asíncronos de propuestas. Cancela esos flujos hijos y espera su salida
real; el loop de voz que los llamó conserva su propia cancelación. El cierre del
observador no oculta la generación nativa. Los jobs posteriores pasan por las
barreras de fuentes vigentes y no retrasan el recibo del conjunto anterior. Los
owners deben registrarse antes de tener trabajo o propuestas pendientes; se
rechaza adoptar silenciosamente un owner que ya estaba activo sin registro.

Los tokens de entrega del `ResponseHandler` son metadatos monotónicos, separados
del texto y de la identidad de hablantes. Permanecen mientras exista controlador,
worker de síntesis o tarea de abortado, incluso cuando la respuesta cancelada ya
volvió al caller. El participante no declara completo ese destino hasta que
terminan los recursos capturados en la prueba. Las sesiones nuevas no reutilizan
esos tokens ni prolongan la espera de los anteriores. El cierre de privacidad
conserva la lease mientras los workers nativos o de voz sigan realmente activos.
Las demás copias de streams y productores siguen en sus participantes propios.

Una prueba detectó una retención adicional: el traceback de un observador
cancelado guardaba el enlace al task nativo y, por él, a su resultado posterior.
La inspección de referencias confirmó ese camino. El frame de `OwnedBlockingWork`
libera ahora sus enlaces a tarea y argumentos al salir; su conjunto de ownership
mantiene el trabajo real hasta terminar. El flujo de propuestas hace lo mismo.
El ensayo conserva al observador cancelado y el recibo pendiente, y comprueba
que el resultado deja de estar alcanzable sin forzar ni desactivar GC.

Los callbacks de disco usan dos threads propios y mantienen su ownership tras
cancelación. Los reintentos de ack conservan el resultado ya terminado sin repetir
la eliminación física. El runtime difiere un conjunto pendiente sin bloquear los
demás destinos; el pool general saturado tampoco impide limpiar el sidecar. La
ruta deshabilitada no importa modelos ni el backend de audio.

Este avance conecta el quinto participante, no un borrado global. Quedan
productores, diagnósticos, colectores, streams y voces, además del resto de H3.
El vencimiento al leer y las barreras de escritura de propuestas ya existen;
el mantenimiento físico de sus TTL sin una solicitud de borrado sigue pendiente.
Tampoco cierra la migración privada, la aceptación de GC y hardware ni H0–H6.
No se modifica producción ni se publica una release.

La [validación del participante de sugerencias](../research/2026-10-26-h3-suggestion-erasure-validation.json)
del commit `ad2386f` registra **4750 passed, 1 xfailed**, sin fallos, en 207,43 s.
Incluye 29 casos nuevos, 807 pruebas dirigidas y 68 comprobaciones finales del
reader, Ruff, Bandit sin hallazgos medios/altos, 18 tests de frontend y smoke del
dashboard. El artefacto compilado verifica recibos y cero filas físicas de
propuestas borradas fuera del checkout en 6,83 s. El heartbeat obtiene p95 de
0,17 ms y máximo de 1,06 ms; la cancelación de control llega a 0,07 ms y el
cleanup a 11,00 ms, con límites intactos.

La evidencia conserva el primer comando de tests corregido por una ruta
inexistente y los dos fallos posteriores: un fixture instalaba demasiado tarde
el fallo de ack, y el resultado nativo seguía enlazado desde el traceback del
observador. El primero se corrige controlando el handoff antes de reconciliar;
el segundo corrige el código de ownership, con inspección de referencias y un
ensayo que no fuerza GC. La suite sin instrumentación no vuelve a medir ni
resuelve las pausas de GC anteriores, ni sustituye la aceptación física de H6.

## Avance: vencimiento físico independiente de propuestas

El participante de sugerencias mantiene un barrido de TTL aunque analytics,
recall y la generación estén apagados. Descubre los sidecars y las fuentes de
eventos registrados, incluidos destinos históricos, sin inicializar loggers,
modelos ni archivos ausentes. No consulta mensajes, patrones ni YAML para
limpiar: sólo metadatos de procedencia, recibos y fechas originales.

Cada lote recorre hasta 32 propuestas y verifica hasta 128 recibos contra las
filas reales de eventos. Una propuesta con más recibos continúa en otro lote;
el cursor conserva un hash de identidad, un offset y fechas, sin arrays de
fuentes ni contenido. Si la fila cambia entre lotes, la verificación comienza
de nuevo. Una fuente inaccesible o de esquema desconocido no prueba un borrado:
queda pendiente hasta poder revisarla o hasta el vencimiento propio verificable.
Una fila desconocida tampoco impide procesar un retiro confirmado posterior.

Se eliminan propuestas vencidas, de eventos retirados o modificados, y de
fuentes cuya revocación ya se confirmó. Los fences de una transacción pendiente
ocultan inferencia, pero no autorizan una eliminación irreversible. El scrub
de texto a las 48 horas no retira los metadatos válidos de actividad.

El máximo propio sigue anclado a la última observación, nunca a una lectura,
reinicio, aceptación o snooze. Un plazo configurado menor se persiste antes de
hidratar consumidores, aunque preceda la antigua fecha de generación. Una
ampliación posterior no restaura ese plazo. La generación rechaza candidatos
ya vencidos antes de llamar al modelo y revalida después de que éste termine.

El mantenimiento tiene un pool nativo propio, dos destinos en paralelo,
reintentos acotados y estados sin rutas o contenido privado. La consulta al
registro no bloquea el barrido de destinos conocidos; un archivo ocupado tampoco
detiene al segundo. Notificaciones de escrituras adelantan el siguiente ciclo;
el polling máximo es de 30 segundos. No promete borrado forense del medio.

El barrido del runtime elimina la copia pendiente del router al vencer y expone
los jobs y scopes de voz que siguen vivos. La entrega interrumpe voz vencida;
un worker nativo no cooperativo conserva ownership hasta salir y no puede
guardar una propuesta tarde. El cierre espera esos recursos y los barridos
reales antes de soltar la lease del catálogo.

Los ensayos dirigidos cubren limpieza sin consultas ni pedidos de borrado,
analytics apagado, reinicios y destinos históricos, plazos reducidos, eventos
retirados o desconocidos, 300 recibos con continuación, cuarentena en varios
lotes, fallos/reintentos, registro y SQLite bloqueados, pool general saturado,
generación tardía y síntesis no cooperativa. El artefacto compilado también
ensaya vencimiento físico después de cerrar el logger productor.

La primera suite completa expuso una carrera previa de capturas: entre un
`SELECT` de archivos vencidos y el `DELETE`, una revocación podía cambiar el
predicado y retirar una fila cuyo localizador no se había encolado. El journal
podía declarar el destino completo mientras quedaba audio sin localizador.
Un ensayo con interleaving controlado reprodujo ese caso; otro reprodujo la
purga irreversible de una revocación pendiente que después hacía rollback.
Ahora `DELETE … RETURNING` devuelve el conjunto exacto retirado y encola sus
archivos en la misma transacción. Los lectores mantienen el fence inmediato,
pero la purga y el borrado explícito exigen revocaciones confirmadas. Ambos
ensayos pasan con la corrección, junto con los demás casos de capturas y TTL.
No se descarta ni se acepta como verde aquella primera ejecución completa.

La [validación del vencimiento](../research/2026-10-26-h3-suggestion-retention-validation.json)
del commit `cba30ec` registra **4782 passed, 1 xfailed**, sin fallos, en 204,35 s.
Incluye 32 casos nuevos, 101 pruebas dirigidas después de la corrección de
capturas, Ruff, Bandit sin hallazgos medios/altos, 18 tests de frontend y smoke
del dashboard. El artefacto compilado pasa fuera del checkout en 6,63 s. El
heartbeat obtiene p95 de 0,50 ms y máximo de 2,52 ms; la cancelación de control
llega a 0,08 ms y el cleanup a 11,85 ms. Los límites y la configuración de GC
permanecen intactos. La evidencia conserva la primera suite no aceptada y los
dos ensayos que fallaban antes de corregir la causa.

Este avance no cierra H3. Siguen pendientes los otros cinco participantes,
consumidores alternos, revisión de voces, pausa/mute y migración real en copia.
La suite sin instrumentación no resuelve la aceptación integrada de GC,
arranque, grandes historiales ni hardware; H0–H6 y la única release final
conservan su alcance. Producción permanece sin cambios.

## Avance: destilación vinculada a fuentes y trabajos reales

La destilación ambiental valida las fuentes concretas antes de construir el
prompt y las vuelve a comprobar durante la inferencia, cada guardado y el ack
del lote. Una revocación o un plazo vencido detiene el modelo cooperativo;
un proveedor que no coopera conserva su scope pendiente hasta salir. Un lote
mixto pierde la derivación completa si se revoca alguna de sus fuentes usadas.
Las filas ajenas o una transacción de revocación que termina en rollback pueden
reintentarse sin marcar una extracción inexistente como completada.

El ciclo libera filas, listas auxiliares y respuestas temporales al salir;
antes de esperar al modelo conserva sólo el prompt seleccionado y su
procedencia. No mantiene las otras filas leídas ni las descartadas por idioma.
El parser distingue un array vacío válido de un error o JSON malformado: sólo
el primero permite confirmar un lote sin hechos. Las respuestas se limitan a
65536 caracteres, 64 hechos y 4096 caracteres por hecho, con confianza finita
entre cero y uno. El helper tolerante anterior conserva su contrato para
consumidores que no confirman trabajo durable.

Cada instancia admite un único ciclo o guardado aún activo. Cancelar al
observador no oculta el guardado: el scope conserva tanto callbacks async
(incluida la delegación nativa de `LongTermMemory.astore_fact`) como adaptadores
síncronos. Estos últimos verifican otra vez las fuentes antes de entrar al
callback, también si esperaron en el pool general. El callback de producción
conserva su guard de catálogo; el scope no sustituye la validación final del
writer ni autoriza un callback externo a ignorar la procedencia recibida.

El builder real vincula la destilación a `PrivacyServices` antes del primer
trabajo. El cierre limita la observación, pero mantiene registrado el drain
real y la lease hasta que terminan modelo y guardados. `AmbientPath.stop`
conserva su cierre aunque se cancele al llamador y cierra el store después de
la destilación real. Detener el job periódico despierta su espera de seis horas
sin necesidad de iniciar otro ciclo.

Se corrige además un enlace en `run_private`: el traceback de cancelar
`asyncio.wait` conservaba su conjunto interno de futures y el resultado tardío
del proveedor. El monitor termina la limpieza y propaga una cancelación nueva,
liberando sus enlaces a operación, tasks, resultado y callback de autoridad.
Mantiene la precedencia de cancelación durante una invalidación y frente a
cancelaciones repetidas. Dos ensayos retienen al observador o a la excepción y
comprueban la liberación del resultado con weakrefs, sin forzar ni desactivar GC.

Los ensayos conectan capturas y catálogo reales, modelo tardío, guardados nativos
y async con delegación real, fallo de commit, cuarentena de respuestas inválidas,
pool general saturado, cierre cancelado y wiring del builder. La prueba del
artefacto compilado construye el mismo builder, guarda un hecho con sus fuentes
y revoca otro lote durante el modelo, sin iniciar micrófonos ni cargar modelos.

La [validación de destilación](../research/2026-10-26-h3-distillation-validation.json)
del commit `acee3be` registra **4812 passed, 1 xfailed**, sin fallos, en 204,66 s.
Incluye 30 casos nuevos, 362 pruebas dirigidas, Ruff, Bandit sin hallazgos
medios/altos, 18 tests de frontend y smoke del dashboard. El artefacto compilado
pasa fuera del checkout en 7,58 s. El heartbeat obtiene p95 de 0,44 ms y máximo
de 2,24 ms; la cancelación de control llega a 0,16 ms y el cleanup a 11,24 ms,
con límites intactos. La evidencia conserva el primer ensayo con cinco fallos:
cuatro aserciones de fixture corregidas y la retención real desde el traceback
de cancelación. La suite sin instrumentación no reevalúa las pausas de GC
anteriores ni acepta por sí sola el runtime integrado.

El participante `PRODUCERS` sigue sin darse por completo: faltan el seguimiento
real de STT, tagger, taps, segmentación y captura de comandos, y su conexión al
journal. Este avance prepara el scope de destilación; no emite un recibo que
prometa haber borrado todos los productores. Los otros cuatro participantes,
voces, pausa/mute, migración y aceptación de runtime/GC y hardware siguen
pendientes dentro del alcance completo H0–H6. Producción no cambia.

## Avance: productores ambientales con seguimiento nativo y retiro de buffers

STT ambiental, lectura de perfiles, identificación de hablante, calidad y DoA
ahora conservan las llamadas nativas reales después de cancelar al observador.
El wrapper de Whisper espera también al worker dedicado del backend. El tagger
devuelve sólo etiqueta y confianza desde el thread: no entrega el embedding
transitorio como resultado de su task. La carga de perfiles sale del loop.

El segmento tiene un scope finito con las fuentes asignadas antes de STT.
`ProducerContext` transporta sólo sala y referencias de captura a las tasks y
workers reales; no concede permisos. Cada etapa comprueba las fuentes en el
handoff. Una llamada que esperó en el pool las vuelve a comprobar antes de
entrar al modelo o leer perfiles. Los threads que ya entraron siguen visibles
hasta salir, aunque el pipeline haya abortado y no vaya a persistir su resultado.
Cancelar internamente el segmento al cerrar no cancela al llamador que sigue
vivo; una cancelación real de ese llamador sí se propaga.

La construcción de cada segmenter, VAD y reset corre en un propietario nativo
serializado por sala. No comparte estado recurrente entre habitaciones. La
carga inicial de Silero se serializa dentro del proceso, porque Torch Hub
modifica una caché común al descargar o instalar el repositorio. Se conserva
la asignación de CPU/GPU existente; estos ensayos no cargan modelos entrenados.

El propietario conserva el lote y su handoff hasta entregar los segmentos o
descartarlos. El preroll, la voz parcial, los segmentos listos y el estado
recurrente tienen fechas observables para el retiro. La tarea de mantenimiento
publica límites cada 50 ms y programa el reset nativo; no espera otra entrada
de audio ni una lectura. Un thread bloqueado durante VAD, construcción o reset
conserva su operación pendiente y la lease del catálogo.

Los límites usan inicio de privacidad, 48 horas desde el input y el fence de la
conversación activa de esa sala. No se aflojan por una publicación atrasada ni
por atrasar el reloj después de observar el vencimiento. La captura automática
no puede asignar una identidad nueva a audio ya retirado. Un fence pendiente
puede descartar estos buffers transitorios, como las cachés privadas; los
registros durables siguen exigiendo un commit para borrarse y un rollback no
los elimina ni confirma un trabajo inexistente.

El tap publica sólo metadatos de admisión al callback de PortAudio. Publicar un
fence o cerrar desacopla la cola antigua: una llamada que ya tenía ese deque
puede perder su chunk, pero no puede reinsertarlo en la cola vigente. El
callback conserva su comportamiento sin locks bloqueantes ni consultas al
catálogo. El cierre rechaza nuevas entradas y espera los trabajos reales antes
de cerrar el store original y soltar la lease. El builder registra esos drains
antes de arrancar; una cancelación del observador no los desregistra.

Los ensayos conectan las clases reales de wrappers, segmenter, tap, catálogo y
SQLite con endpoints nativos sintéticos. Incluyen las cinco etapas retenidas,
el executor dedicado de Whisper, VAD y construcción bloqueados, reset tardío,
expiración sin entrada, otra sala viva, rollback, saturación del pool, carreras
del callback y weakrefs a audio y resultados después de cancelar, sin manipular
GC. El artefacto compilado construye el builder real, alimenta el propietario
de segmentación y comprueba una inferencia nativa pendiente tras revocar y
después de agotar la observación de cierre.

La inspección posterior reprodujo otra intercalación del tap: leer primero el
fence viejo y después la cola nueva permitía reinsertar un chunk retirado.
La primera suite completa se interrumpió tras esa reproducción y no se acepta
como verde. El callback ahora toma la cola antes del fence; la publicación
instala el fence más estricto antes de sustituir el deque. Los ensayos cubren
ambos órdenes y verifican que la entrada fresca sigue funcionando.

ASR y el identificador compartidos serializan también su entrada nativa para
evitar inicializaciones lazy o inferencias simultáneas sobre el mismo backend.
Un waiter vuelve a comprobar su captura cada 50 ms: si se revoca, libera sus
inputs sin esperar a que termine otra sala viva. El callback de audio no usa
estos locks; sólo los workers nativos esperan en ellos.

La [validación de productores ambientales](../research/2026-10-26-h3-ambient-producers-validation.json)
del commit `cdc27bf` registra **4848 passed, 1 xfailed**, sin fallos, en 197,88 s.
Incluye 36 casos nuevos, 520 pruebas dirigidas antes de las últimas correcciones
y 296 después, Ruff, Bandit sin hallazgos medios/altos, 18 tests de frontend y
smoke del dashboard. El artefacto compilado pasa fuera del checkout en 7,67 s.
El heartbeat obtiene p95 de 0,21 ms y máximo de 1,05 ms; la cancelación de control
llega a 0,10 ms y el cleanup a 11,95 ms, sin ampliar límites ni manipular GC.

La evidencia conserva el ensayo de wake ajustado para continuar de forma
deliberadamente no cooperativa, el fallo de propagación de cancelación al
padre y su corrección, y la reproducción del callback que reinsertaba un chunk.
La ejecución completa interrumpida por ese hallazgo queda fuera de aceptación;
sus errores de teardown de pytest corresponden a esa interrupción. La suite
verde posterior no reevalúa las pausas de GC medidas antes ni sustituye los
ensayos de hardware, recuperación y arranque de H4/H6.

`PRODUCERS` sigue abierto: faltan los buffers y jobs previos al procesador de
comandos, wake VAD, STT temprano y consumidores de aprendizaje, y conectar los scopes completos al
journal. No se anuncia borrado global ni se habilita la UI de borrado amplio.
Los otros participantes, revisión de voces, pausa/mute, migración y las
aceptaciones físicas e integradas de H0–H6 siguen pendientes. La suite sintética
no resuelve las pausas de GC previamente medidas ni la recuperación real de H4.
Producción permanece sin cambios.

### Propietario nativo del procesador de comandos

El `CommandProcessor` conserva las operaciones reales de STT primario y sombra,
lectura de perfiles, identificación y emoción. Las rutas secuencial, paralela y
con transcripción previamente verificada ejecutan esos trabajos fuera del bucle
principal. La llamada al detector de emociones usa su firma real también con
texto reutilizado; los endpoints sintéticos anteriores no bastaban para probarla.

Ambos routers heredan la procedencia del capture antes de entrar al procesador,
también sin memoria semántica. El monitor cancela el controlador al vencer o
revocar una fuente; los trabajos nativos siguen visibles hasta salir. La entrada
al modelo compartido dentro del procesador se serializa y vuelve a comprobar la
fuente cuando espera. Los resultados tardíos se descartan y el estado de usuario
y emoción del último comando comprueba la misma fuente al leer y al mantenerlo.

El veto conserva la distinción entre silencio primario y descarte por sombra;
error o timeout del shadow siguen permitiendo el texto primario. Su timeout limita
la observación, sin cancelar el trabajo real. El modo observador conserva su
propietario y monitor después de responder. Los logs del procesador registran
presencia de texto y latencia, sin transcripciones, nombres ni traceback nativo.
Esto no resuelve los demás destinos de `DIAGNOSTICS`.

`main` inyecta la autoridad común y registra el cierre al construir el componente.
El warmup del `VoicePipeline` usa el mismo propietario; el cierre de la autoridad
espera su salida efectiva aunque venza el plazo del observador. Los ensayos usan
catálogo real, ambos routers, el warmup del pipeline, el executor dedicado real de
Whisper y el adaptador real de emoción, con endpoints CPU sintéticos. Cubren
revocación, TTL, rollback del marcador, espera de modelo y executor, cancelación,
cierre tardío y liberación de audio/resultados con weakrefs, sin manipular GC.

`PRODUCERS` permanece abierto: aún faltan los buffers de captura de comandos,
wake/VAD, STT temprano, consumidores de aprendizaje y su integración completa
con el journal. Este cambio no permite anunciar borrado global ni habilitar la
UI de borrado amplio. Tampoco sustituye las aceptaciones de hardware, runtime,
GC y recuperación pendientes.

La [validación de comandos](../research/2026-10-26-h3-command-producers-validation.json)
del commit `8ff3076` registra **4880 passed, 1 xfailed**, sin fallos, en 207,04 s.
Incluye 32 casos nuevos conectados y 625 pruebas dirigidas con optimizaciones
antes de los tests de audio. Pasan Ruff, Bandit sin hallazgos medios/altos,
18 tests de frontend y el smoke del dashboard. El artefacto sin fuentes verifica
sus 314 hashes y ejercita ambos cierres nativos fuera del checkout en 2,22 s.
El heartbeat obtiene p95 de 0,23 ms y máximo de 1,21 ms; la cancelación de control
llega a 0,08 ms y el cleanup a 11,20 ms. Se conservan límites y GC normal.

La primera suite completa detectó dos usos del helper paralelo eliminado;
se restituye su contrato y la ruta de producción lo usa con los mismos trabajos
nativos retenidos. La verificación dirigida encontró además que el test de
warmup de Piper dejaba dos módulos distintos según el orden de ejecución;
ahora reemplaza sólo la librería importada de forma perezosa, sin recargar el
módulo de la aplicación. La evidencia conserva esos fallos y sus correcciones.
No se han vuelto a medir ni corregido las pausas globales de GC observadas antes.
La producción y los datos privados reales permanecen sin cambios.

### Captura multiroom antes del procesador

El callback de PortAudio conserva telemetría y tee ambiental, y copia los frames
a una cola acotada por sala. No ejecuta wake, VAD ni consultas al catálogo.
Cada sala tiene un propietario que serializa inferencia, buffers, endpoint,
snapshots de STT temprano, publicación de hints y reset fuera del bucle principal.
Las otras salas pueden seguir capturando si una inferencia queda bloqueada.

La procedencia se crea antes de entrar a wake, desde el frame más antiguo del
lote. El evento conserva esos IDs y deadline al pasar al procesador, sin renovar
el plazo ni atribuir al dispatch el momento de captura. El estado del modelo
forma parte de esa procedencia: su epoch es conservador y se retira como máximo
al pasar una hora sin captura activa, o al vencer o revocar su fuente. Por eso
un comando puede heredar un plazo anterior al de su fragmento final. El preroll,
las ventanas de Whisper y el estado recurrente se resetean antes de reutilizar
el modelo bajo otra fuente. Los huecos de entrada largos también retiran el
estado anterior; un frame previo al epoch no se renombra como entrada nueva.

El callback usa deques desacoplables con hasta 32 frames; el propietario consume
lotes de hasta ocho cada 50 ms. La publicación del fence precede al cambio de
cola. Un callback antiguo puede perder su frame, pero no repoblar la cola actual.
El retiro de un stream publica también un fence local antes de desacoplar su
cola; el callback vuelve a comprobar que conserva el holder vigente. Esto
impide que una llamada pausada reaparezca como entrada nueva al avanzar el
reloj. Los publicadores de control serializan sus límites, sin locks en
PortAudio, para que una observación atrasada no afloje ese fence.
El mantenimiento retira estado incluso sin nuevos frames. Los trabajos nativos
y resets siguen visibles después de cancelar al observador y retienen la misma
autoridad de privacidad hasta salir. El handle de Porcupine se elimina, porque
su API no ofrece un reset del historial interno. El siguiente frame admitido
lo recrea de forma perezosa.

El STT temprano comparte admisión de modelo con wake y el procesador principal;
vuelve a comprobar fuente, generación y cancelación al esperar, antes de inferir
y antes de publicar. Un snapshot anterior no sobrescribe una captura siguiente.
La apertura por barge-in publica `listening` después de preparar su continuación.
Los dispatches pendientes se acotan y el seguimiento de sus callbacks conserva
el cleanup real. `main` registra el cierre al construir el loop; el warmup se
ejecuta en trabajo nativo retenido y un fallo de wake se limita a esa sala.

Este avance no completa `PRODUCERS`: siguen pendientes el camino alternativo
`AudioLoop`/`AudioManager`, los consumidores de clips y aprendizaje, la integración
completa al journal y las garantías de los streams físicos. Tampoco completa
`DIAGNOSTICS`, `COLLECTORS`, `STREAMS`, `VOICES`, pausa/mute, migración ni la
aceptación integrada de hardware y runtime/GC. No se anuncia borrado global ni
se habilita la UI de borrado amplio. La validación completa acompaña al commit
construido y probado, antes de cualquier release.

La [validación de captura](../research/2026-10-26-h3-capture-producers-validation.json)
del commit `71c0588` registra **4908 passed, 1 xfailed**, sin fallos, en 201,40 s,
con 28 casos nuevos y 814 pruebas dirigidas. Pasan Ruff, Bandit sin hallazgos
medios/altos, 18 tests de frontend y el smoke del dashboard. El artefacto sin
fuentes verifica 315 hashes y ejecuta el `run` real con InputStream sintético,
endpoint, router y procesador, incluyendo la procedencia original, retiro del
driver, revocación y cierre retenido hasta salir del modelo, en 2,63 s.

La primera suite de borrador pasó, pero una reproducción independiente
reintrodujo un frame desde un callback retirado: no se acepta ese borrador.
Las correcciones del fence, holder y publicación atrasada quedan en `d966ff9`.
La siguiente suite detectó que un helper de prueba cancelaba el cleanup del
coordinador al ser cancelado por el monitor de fuentes. El catálogo tenía el
tombstone y el recibo pendiente con error; no hubo una confirmación falsa de
borrado. El helper ahora observa con `shield`, y una compuerta demuestra que
cancelar ese observador no cancela la tarea real. La suite final verifica también
la retirada efectiva de la fila y del FLAC. La evidencia conserva ambos hallazgos,
la carrera de publicación de barge-in y la corrección del fixture RMS sintético.

El heartbeat llega a p95 de 0,27 ms y máximo de 1,09 ms; la cancelación de control
alcanza 0,18 ms y el cleanup 11,31 ms, sin ampliar límites ni manipular GC.
Esto no vuelve a medir ni corrige las pausas globales de GC observadas antes;
la aceptación integrada y física de H4/H6 sigue pendiente. No se modificaron
producción ni los datos privados reales.

## Avance: captura de entrada única y detector ambiental opcional

El camino alternativo `AudioLoop`/`AudioManager` usa ahora el actor de captura
compartido. El callback sólo copia a la cola acotada; wake, endpoint y reset
se ejecutan fuera del event loop con comprobaciones de fuente antes y después
del trabajo real. El timer conserva el momento de adquisición del frame, aunque
el worker lo procese más tarde. La entrega de comandos y el análisis ambiental
tienen admisión acotada e independiente de la captura; un consumidor lento no
detiene la lectura. `main` inyecta la autoridad común y registra el cierre antes
del warmup.

La construcción, apertura, parada y liberación del driver tienen propietario
nativo. Cancelar al observador no pierde un handle creado tarde: se retiran sus
callbacks, se espera la apertura real y se reintenta el cierre sobre ese mismo
handle. Un error del hook de retiro tampoco omite la liberación. La lease de
privacidad permanece hasta que terminan driver, modelo y callbacks reales.
El estado público sólo muestra contadores, sin handles ni contenido privado;
los errores nuevos registran el tipo, sin copiar el payload de una excepción.

Las ventanas ambientales conservan todas sus fuentes originales y el plazo más
antiguo. Los parciales vencidos se retiran incluso sin nuevos frames; una pausa
de entrada no se empalma como audio continuo. El `AudioEventDetector` opcional
retiene carga, inferencia, features y callbacks síncronos fuera del loop; vuelve
a comprobar las fuentes antes de cada notificación. La onda se libera al acabar
la inferencia, aunque una notificación siga pendiente. Los callbacks asíncronos
no cooperativos conservan su propietario hasta terminar y no permiten una
entrega posterior desde fuentes revocadas. Un callback puede solicitar su propio
cierre sin esperarse a sí mismo.

La validación dirigida registra 702 pruebas pasadas, incluidos 32 casos nuevos
con driver, manager, detector y catálogo reales y fronteras nativas sintéticas.
La primera suite completa encontró dos timeouts al preparar filas para pruebas
de borrado de eventos, antes de pedir el borrado. Ese montaje esperaba al
observador de 100 ms como si garantizara la escritura física. Ahora espera la
escritura real en su propietario; las pruebas del observador acotado siguen
usando la misma API y el mismo límite. Un caso adicional con compuertas comprueba
el timeout original, la escritura tardía retenida, la preparación independiente
del plazo del observador y el borrado físico de ambas filas. La evidencia
conserva la suite rechazada; no atribuye estos timeouts a GC sin instrumentación.
La [validación de entrada única](../research/2026-10-26-h3-single-input-validation.json)
del commit `03c2f08` registra **4941 passed, 1 xfailed**, sin fallos, en 204,05 s.
Incluye los 32 casos de captura y la regresión de preparación de eventos;
Ruff, Bandit sin hallazgos medios/altos, 18 tests de frontend y smoke del
dashboard pasan. El artefacto sin fuentes verifica 316 hashes y ejecuta
`AudioLoop.run`, driver, wake, endpoint, router, procesador y detector opcional
con fronteras sintéticas, fuera del checkout, en 2,34 s. Comprueba fuentes
originales y mezcladas, liberación de la onda antes de notificar y lease retenida
hasta que terminan callback y cierre del driver.

El heartbeat alcanza p95 de 0,24 ms y máximo de 1,04 ms; la cancelación de
control llega a 0,06 ms y el cleanup a 11,13 ms. No se ampliaron presupuestos ni
se manipuló GC. Estas pruebas no vuelven a medir ni resuelven las pausas globales
observadas antes, ni sustituyen la aceptación física e integrada de H4/H6.
Esto cubre ambos caminos de captura, pero siguen pendientes los consumidores
de clips/aprendizaje, la integración completa de productores al journal y el
ciclo físico del driver multiroom. No completa `PRODUCERS`, `STREAMS` ni H3.

## Avance: clips de wake con fuentes y localizadores durables

`WakeClipWriter` recibe ahora un store preparado con la autoridad común. Cada
trabajo copia su onda, conserva las fuentes originales y el menor vencimiento,
y rechaza entradas sin fuentes o revocadas. Los wakes aceptados/rechazados
heredan el contexto del actor de captura; el wake textual transmite explícitamente
la fuente ambiental al callback de clips missed, sin inventar otra captura.
Los archivos nuevos usan UUID, sin sala ni score en el nombre, y permisos 0600.
Sala, score, bucket y fuentes quedan en el catálogo privado y vencen con el clip.

El catálogo guarda el intento antes de crear un `.part`. La codificación corre
en el hilo retenido; el rename final y su recibo comparten la barrera de admisión
del catálogo. Una revocación anterior impide publicar; una posterior encuentra
el localizador. Se eliminan parte y WAV antes de soltarlo. Un unlink fallido
conserva el intento para reintento y reinicio; los paths se recorren mediante
descriptores sin seguir symlinks reemplazados. La rotación sólo alcanza clips
gestionados y mantiene topes independientes por bucket. Los WAV legacy quedan
fuera de esta política y no se borran ni se importan por proximidad de nombre.

La retención se construye antes del writer y sigue activa con la captura de
clips apagada. Examina hasta 128 filas por trabajo, conserva las raíces
históricas registradas y despierta por cambios o por el plazo más cercano. La
cola se purga cada 50 ms sin esperar a un encoder bloqueado. Un vencimiento
observado de un trabajo queda invalidado aunque retroceda el reloj.
Cerrar la captura o privacidad descarta la cola y espera al hilo real; un
timeout del observador no libera la lease. El cierre del pool privado tampoco
depende de un executor general saturado. Un diagnóstico bloqueado después de
fallar no conserva la onda ni el traceback privado del trabajo terminado.

Este avance registra destinos de collectors, pero no da de alta un driver que
finja completar todo `COLLECTORS`. Faltan conversaciones, correcciones STT y
otros datasets de aprendizaje, además del journal completo de productores.
La limpieza física local de clips no constituye un recibo global de borrado.
La [validación de clips de wake](../research/2026-10-26-h3-wake-clips-validation.json)
del commit `a239fa7` registra **4971 passed, 1 xfailed**, sin fallos, en 206,89 s.
Incluye 30 casos nuevos y los 11 casos de formato, rotación y cola adaptados al
contrato con fuentes. Pasan 876 pruebas dirigidas, los 41 casos finales del
writer/privacidad, Ruff, Bandit sin hallazgos medios/altos, 18 tests de frontend
y smoke del dashboard. La prueba de saturación expuso inicialmente el cierre
delegado al pool general; la evidencia conserva ese fallo y su corrección,
sin ampliar límites. También comprueba la liberación de la onda antes de un
diagnóstico bloqueado, sin forzar GC.

El artefacto sin fuentes verifica 317 hashes y ejecuta el warmup y `run` reales
multiroom con InputStream y modelos sintéticos, endpoint, router, procesador y
writer. Comprueba intención durable y fuentes antes de codificar, revoca con
el encoder bloqueado, mantiene la lease durante los timeouts y, al liberar la
compuerta, rechaza el WAV tardío y retira parte, WAV y metadatos. También espera
la salida real del hilo y del pool privado, en 2,66 s fuera del checkout.

El heartbeat llega a p95 de 0,18 ms y máximo de 1,02 ms; la cancelación de
control alcanza 0,08 ms y el cleanup 11,17 ms. Los presupuestos siguen intactos.
Esta suite no vuelve a medir las pausas globales de GC anteriores ni cierra la
aceptación física e integrada de H4/H6. No se tocaron producción ni datos reales.

## Avance: correcciones STT con audio y texto bajo la misma fuente

`STTCorrectionCollector` comparte ahora el propietario de archivos y la autoridad
de privacidad con los clips de wake. Exige fuentes para el audio/transcripción
original y para el feedback; el contexto de la interacción puede aportar las
segundas, pero nunca se inventan las primeras. El registro conserva ambas
procedencias y el vencimiento más antiguo. El WAV usa UUID y permisos 0600;
el texto original, la corrección y el identificador del usuario permanecen en
el catálogo, vinculados al localizador por una FK con borrado en cascada.
No se generan nuevos `metadata.json` ni se leen los legacy para entrenar.

La admisión nativa está acotada a ocho operaciones. Las ondas se copian antes
del handoff; si vencen en la cola del executor se retiran aunque los dos
encoders sigan bloqueados. Un barrido físico pendiente ya no detiene esa purga:
su observador se ejecuta aparte del controlador de buffers. El vencimiento se
mantiene invalidado tras un retroceso del reloj. Cancelar al solicitante o
agotar su espera de cierre conserva el trabajo real y la lease hasta terminar;
un error de escritura no copia texto privado a logs ni conserva su payload
mientras un diagnóstico está bloqueado.

Las lecturas devuelven vistas revocables acotadas. Cada campo raw comprueba
fuentes y plazo al acceder; la purga independiente también lo retira sin una
lectura posterior. El cierre, el borrado del archivo y la rotación invalidan
las vistas ya entregadas. El límite de 256 vistas retiene sólo las activas y
libera admisión al vencer. Marcar una corrección como usada compara recibos
reales, no paths aportados por el caller, y no renueva la fecha. La retención
manual sólo puede acortarla a 48 horas o menos.

Un acortamiento se propaga a las vistas y trabajos del mismo destino antes
de su commit. Los fences compartidos sólo se ajustan a plazos menores; un
lector de otra instancia no puede conservar el plazo anterior. El publish
consulta además su localizador durable y vuelve a comprobar el plazo real
después del rename/fsync, aunque la fuente original todavía siga vigente.
Las pruebas con encoder y rename bloqueados reprodujeron publicaciones tardías
que una primera suite verde no había cubierto; ese borrador se conserva como
no aceptado. La corrección también cubre la vista ya entregada a otro colector.

`main` construye el colector configurado con ese mismo store. No se agrega un
nuevo productor de correcciones ni una sesión de entrenamiento. El consumidor
Whisper antiguo escribe datasets sin procedencia y no puede retirar capturas
de un adapter: se bloquea para el colector privado antes de consultar datos,
escribir JSONL, generar scripts o lanzar un proceso. El estado de ese paso
queda bloqueado; las pruebas de los adaptadores sintéticos anteriores siguen
separadas. Esto no desarrolla LoRA nuevo ni valida sus recursos para H6.

La retención registra raíces STT actuales e históricas aunque la función esté
deshabilitada. Una codificación incompleta mantiene parte y localizador hasta
el unlink; al retirarlo se elimina también el texto vinculado. Los archivos
legacy quedan en cuarentena y los desconocidos no autorizan una lectura.
Faltan las conversaciones y otros datasets, productores y journal completo;
`COLLECTORS` continúa sin un driver que finja borrado global. La validación
de [correcciones STT](../research/2026-10-26-h3-stt-corrections-validation.json)
del commit `042ebbb` registra **4998 passed, 1 xfailed**, sin fallos, en 208,90 s.
Incluye 27 casos nuevos y los siete casos originales de formato, validación,
conteo, lectura y marcado adaptados a la autoridad. Pasan 358 pruebas dirigidas,
Ruff, Bandit sin hallazgos medios/altos, 18 tests de frontend y smoke del
dashboard. La evidencia conserva el borrador verde rechazado, las tres
reproducciones de plazos acortados y el bloqueo de la purga detrás del pool,
con sus correcciones. No se ampliaron presupuestos para aceptarlas.

El artefacto sin fuentes verifica 317 hashes y ejecuta captura multiroom,
endpoint, router, procesador y el factory del colector sobre audio y modelos
sintéticos. Guarda una corrección con fuentes originales y de interacción,
bloquea la exportación antigua, acorta su plazo desde otro colector y comprueba
que la vista vence mientras sus fuentes originales siguen vivas. Con otra
codificación retenida verifica localizador y plazo durable, revoca el feedback,
conserva la lease durante el timeout y, al liberar el encoder, retira parte,
WAV y texto en cascada. Termina fuera del checkout en 2,74 s.

El heartbeat llega a p95 de 0,17 ms y máximo de 1,01 ms; la cancelación de
control alcanza 0,06 ms y el cleanup 10,93 ms. Esta suite no vuelve a medir las
pausas globales de GC observadas antes ni cierra la aceptación física e
integrada de H4/H6. Producción y los datos privados reales siguen sin cambios.

## Avance: turnos y exportaciones de conversaciones

`ConversationCollector` ya no carga `conv_*.json` ni mantiene otra copia raw
global en `_conversations`. Cada turno nuevo exige fuentes explícitas del
input y de la respuesta completa. Se publica como JSON inmutable con UUID,
hash, permisos 0600 y localizador durable; grupo, sala y calidad se indexan en
el catálogo. Su plazo es el menor de todas las fuentes, sin renovarlo al leer,
agrupar, marcar o exportar. Las vistas de turnos comprueban sus recibos al
acceder y se purgan sin una lectura posterior.

El feedback reemplaza el turno por otra versión inmutable. La transacción
comprueba que la versión anterior sigue vigente, publica la nueva y retira la
anterior. La marca incorpora su propia fuente y conserva el plazo original.
El feedback por voz busca el último turno de la misma sala y conversación;
un caller con una vista real puede adjuntar una interacción explícita. Los
nombres son sólo datos raw y no conceden permisos. El router espera el recibo
real y no anuncia una marca guardada si no existe un turno vigente.

Las exportaciones Alpaca, ShareGPT y conversaciones son JSONL temporales en
destinos registrados. Cada entrada conserva fuentes y el turno original; el
archivo completo vence con el plazo más antiguo. Los lotes se limitan por
cantidad y por las 64 fuentes del scope; un cursor permite continuar sin
truncar procedencia. Los edges de dependencia permanecen aunque se retire el
turno padre: cambiar su calidad o borrarlo invalida las vistas, bloquea un
exporter retenido y permite retirar el archivo derivado después de reiniciar.
El intento precede al `.part`; los errores de unlink conservan el localizador.

La publicación vuelve a comprobar el plazo después del callback transaccional
y del commit real de SQLite. Las compuertas reprodujeron ambos vencimientos
tardíos antes de corregirlos. La transferencia de procedencia al reemplazo se
valida dentro de la transacción; retirar el predecesor no cancela la versión
nueva que acaba de autorizarse. Ocho operaciones y 256 vistas acotan el trabajo
y las copias retenidas. El cierre conserva recursos nativos reales y descarta
payloads de cola sin esperar a encoders o barridos ocupados.

El CLI de datos abre la autoridad privada, consulta estadísticas sin cargar
texto legacy y devuelve direcciones temporales de exportación. Los consumidores
de hábitos y entrenamiento antiguos rechazan el colector privado antes de
copiar su texto; LoRA rechaza estos exports gestionados antes de cargar modelos.
El perfil candidato deja el entrenamiento nocturno deshabilitado: sus copias y
recursos aún no cumplen la aceptación integrada de H6. No se instala ni se
reinicia producción y no se desarrolla LoRA nuevo. `main` construye el colector
para retención y feedback independientemente de ese flag; este paso no agrega
un productor automático de turnos.

Siguen pendientes otros datasets y consumidores legacy, journal completo,
diagnósticos, streams, voces y los demás entregables de H0–H6. Un turno o export
retirado localmente no equivale a un recibo global de `COLLECTORS`.
La [validación de conversaciones y exports](../research/2026-10-26-h3-conversation-collector-validation.json)
del commit `fc04490` registra **5034 passed, 1 xfailed**, sin fallos, en 211,46 s.
Incluye 36 casos nuevos y 815 pruebas dirigidas; pasan Ruff, incluido el CLI,
Bandit sin hallazgos medios/altos, 18 tests de frontend y smoke del dashboard.
La evidencia conserva las dos reproducciones de commit tardío y el fixture
del CLI que había sembrado capturas futuras respecto de su reloj real. Las
correcciones no amplían límites ni relajan validación de fuentes.

El artefacto sin fuentes verifica 317 hashes y ejecuta captura, endpoint,
router y procesador reales con modelos y audio sintéticos. El feedback pasa por
el router y espera la nueva versión; corregirla invalida su export anterior
aunque las capturas sigan vivas. Con un encoder retenido verifica lease y
descarte tardío; un arranque nuevo sin entrenamiento retira los JSON y JSONL
vencidos. El ejercicio aceptado dura 1,12 s y reutiliza el mismo artefacto.

La configuración permanece fuera del paquete, como exige el builder. La
prueba verifica el flag nocturno desde la revisión exacta de Git y suministra
solamente ese perfil saneado junto al artefacto; no copia configuración privada
al tar. También conserva los ajustes de fixture del gate acústico y del
endpoint sintético de respuesta. Nada de esto valida todavía la configuración
privada instalada ni sustituye la integración de H6.

El heartbeat alcanza p95 de 0,39 ms y máximo de 1,16 ms; la cancelación de
control llega a 0,06 ms y el cleanup a 11,03 ms. No se manipuló GC. Las pausas
globales anteriores, el hardware y la aceptación integrada siguen pendientes.
Producción y los datos privados reales permanecen sin cambios.

## Avance: admisión del entrenamiento antiguo

El rechazo de conversaciones privadas ocurría después de construir el generador
de hábitos y el entrenador. Sus constructores todavía podían cargar perfiles e
historial legacy y crear directorios antes de llegar a esa comprobación. Una
reproducción sobre `3c5a934`, con archivos sintéticos, confirma ambas lecturas y
la creación del directorio de adapters; no inicia scheduler ni carga modelos.

Los constructores que reciben la autoridad privada o dependencias gestionadas
quedan ahora en estado bloqueado antes de consultar el filesystem. `main`
inyecta explícitamente la autoridad aunque una dependencia esté deshabilitada.
Habilitar el flag nocturno no inicia un thread ni anuncia que quedó programado.
Una ejecución manual devuelve `blocked` con un motivo fijo, sin descargar
modelos, invocar callbacks, exportar datasets, guardar historial ni generar
reportes. La consulta de adapters tampoco abre archivos legacy. Los métodos de
hábitos y los accesos directos de datasets/procesos comprueban la admisión antes
de usar incluso ejemplos ya cacheados.

Este bloqueo conserva los archivos originales y mantiene las pruebas de
adaptadores sintéticos. No incorpora entrenamiento privado ni sanea los datasets
anteriores: siguen pendientes su inventario, cuarentena, retención y los otros
consumidores alternos. El CLI nocturno sin autoridad se corrige en el avance
siguiente. Tampoco completa `COLLECTORS`, los demás participantes del journal
o H6. La configuración instalada y producción permanecen sin cambios.

La [validación de admisión](../research/2026-10-26-h3-private-training-admission-validation.json)
del commit `b793903` registra **5050 passed, 1 xfailed**, sin fallos, en 205,88 s.
Incluye 16 casos nuevos y 129 pruebas dirigidas. Pasan Ruff, Bandit sin hallazgos
medios/altos, 18 tests de frontend y el smoke del dashboard. El único xfail sigue
siendo la limitación conocida de elipsis del extractor NLU.

El artefacto sin fuentes verifica 317 hashes e importa el `main` compilado fuera
del checkout. Con autoridad SQLite y factories reales de ambos colectores,
fuerza un perfil sintético con el flag nocturno habilitado: no consulta perfiles
ni historial legacy, no crea threads ni datasets y no llama modelos o procesos.
Una ejecución manual devuelve `blocked` y los archivos originales permanecen
idénticos. El ejercicio dura 4,11 s; la configuración sigue fuera del paquete.
Esto no ejecuta el arranque completo de `main` ni prueba hardware o recursos de
entrenamiento.

El heartbeat registra p95 de 0,28 ms y máximo de 1,07 ms; la cancelación de
control llega a 0,07 ms y el cleanup a 10,86 ms, con los límites anteriores.
No se manipuló GC. Las pausas instrumentadas previas y la aceptación física e
integrada de H4/H6 continúan pendientes.

## Avance: CLI nocturno y entrada directa de LoRA

Una reproducción sobre `8a6864d` confirma dos accesos alternos: el CLI nocturno
construía un entrenador sin autoridad y exponía historial y paths de adapters;
el filtro de LoRA por directorio rechazaba el export canónico, pero una copia
del mismo JSONL alcanzaba la importación del backend. La reproducción usa sólo
datos sintéticos e intercepta esa importación antes de cargar modelos.

El CLI nocturno ya no construye un entrenador para ninguna de sus acciones.
`--status` devuelve disponibilidad bloqueada y aclara que no consultó el runtime;
`--collect` y `--run` devuelven un motivo fijo y exit code 1. La opción histórica
de GPUs no asigna recursos. La ayuda y los errores de selección tampoco abren
archivos. El CLI de LoRA rechaza entrenamiento y listado antes de construir un
trainer o copiar argumentos privados a la salida.

La entrada directa de `LoRATrainer` también queda bloqueada para cualquier path,
incluidos copias, symlinks y hardlinks de exports reales. El constructor no
consulta ni crea archivos y el listado no abre adapters sin una sesión
admitida. Se conserva la implementación antigua para revisión; no se agrega un
flag para habilitarla, entrenamiento nuevo ni una reasignación de GPUs. El CLI
de estadísticas y exports temporales del catálogo sigue disponible.

Esto cierra los bypasses identificados de esos accesos, no la retención física
de datasets y adapters históricos ni el resto de `COLLECTORS`. La
[validación conjunta de CLI y cierre](../research/2026-10-27-h3-training-cli-validation.json)
del commit `4f82a85` registra **5073 passed, 1 xfailed**, sin fallos, en 207,59 s,
con 19 casos nuevos de CLI y cuatro de cierre. Pasan las 177 pruebas dirigidas
de entrenamiento, las 51 dirigidas al cierre, Ruff, Bandit incluyendo el CLI
sin hallazgos medios/altos, 18 tests de frontend y el smoke del dashboard.
Producción y configuración instalada permanecen sin cambios.

## Avance: cierre de memoria con callbacks pendientes

La primera suite completa del cambio de CLI, sobre `0fb14ee`, se detuvo en
privacidad transitoria mientras consumía un núcleo. Se conservó como no
aceptada y se interrumpió para obtener el stack: el cierre de `LongTermMemory`
iteraba sobre tareas terminadas dentro de `gather`. En CPython 3.13 ese gather
puede terminar inmediatamente, sin ejecutar los callbacks que retiran las
tareas del conjunto; el bucle impedía incluso procesar su propio timeout.

Una reproducción aislada ordena la finalización del trabajo justo antes del
cierre y deja el callback de retirada detrás de él. El código anterior agota
un watchdog externo de tres segundos; el corregido termina y retira la tarea.
El cierre usa `wait` para permitir esos callbacks y conserva la espera del
trabajo nativo real. Cuatro casos en procesos separados cubren resultado,
error, cancelación y un worker nativo bloqueado: el timeout de observación
mantiene la lease hasta que termina ese worker. No se fuerzan ni deshabilitan
GC ni sus pausas, y no se amplían límites de runtime para aceptar el cambio.

La suite aceptada del commit `4f82a85` incorpora los cuatro casos y completa
también privacidad transitoria, donde se detuvo la primera ejecución. La
evidencia conserva el exit code 134 de la interrupción diagnóstica, su stack y
la reproducción con watchdog externo; no presenta aquella suite como verde.
La prueba de CLI anterior no cubría este cruce de cierre.

El nuevo artefacto verifica 317 hashes y ejecuta los cuatro casos de cierre
desde bytecode, incluida la lease retenida por un worker real. También ejecuta
siete procesos CLI: los accesos nocturnos y LoRA quedan bloqueados, mientras las
estadísticas y los exports gestionados siguen funcionando con autoridad SQLite,
turnos y exports reales sobre texto sintético. Conserva sin cambios los archivos
legacy. El ejercicio dura 2,03 s fuera del checkout y mantiene configuración,
datos y modelos fuera del paquete. No ejecuta el arranque completo de `main`
ni prueba hardware o entrenamiento.

El heartbeat alcanza p95 de 0,21 ms y máximo de 1,11 ms; la cancelación de
control llega a 0,08 ms y el cleanup a 11,09 ms. Los presupuestos se conservan.
Corregir este bucle no equivale a resolver las pausas globales de GC observadas
antes: su aceptación integrada y la del hardware continúan abiertas.

## Avance: ensayo privado de la selección legacy en copia

La vista de H0 contenía sólo metadata, sin backup de documentos ni vectores.
La lectura actual observa 874 registros en la colección seleccionada: siguen
presentes los 870 IDs de esa vista y hay cuatro nuevos. El ensayo selecciona
exactamente los 870 originales; no amplía implícitamente la migración.

`scripts/stage_legacy_memory.py` prepara un directorio nuevo y privado. Copia
SQLite mediante una conexión `mode=ro` y `query_only`, verifica la integridad del
snapshot y conserva hashes del origen observados antes/después. No abre Chroma
sobre el origen ni escribe allí. Guarda manifest, snapshot y catálogo candidato
con permisos restrictivos, sin sobrescribir un destino existente. La selección
se comprueba en lotes acotados y la copia/importación tiene un presupuesto de
trabajo; el cierre conserva la propiedad nativa hasta que termina.

El catálogo candidato conserva documentos, metadata tipada y vectores FLOAT32
en tablas de cuarentena separadas, dentro de una sola transacción con lease.
No crea hechos, propuestas, fuentes, confirmaciones ni jobs de indexación. Las
fechas de inserción antiguas siguen siendo metadata legacy, no fechas de
extracción. No se inventa autor o conversación. Los registros quedan fuera de
los lectores normales y de la confirmación de propuestas, incluso con un índice
contaminado que afirma falsamente que fueron confirmados.

Reanudar verifica el mismo snapshot y los mismos registros, sin abrir el origen
ni renovar la retención. Una fila retirada o cambiada causa rechazo; un retry no
la recrea. Una transacción fallida se revierte completa. También se rechazan
links en el candidato/lease y orígenes que no son archivos regulares. El CLI
devuelve contadores o un error fijo, sin documentos ni IDs originales.

El ensayo privado real pasó sobre los 870 IDs, conservando sus 870 documentos
y vectores; los cuatro registros nuevos no se importaron al catálogo candidato.
Los hashes de SQLite/WAL observados antes y después coinciden. Reanudar no
duplica filas y dos arranques del grafo de memoria conservan 870 cuarentenas y
cero hechos, fuentes, propuestas, resultados de búsqueda y jobs de indexación.
El directorio es 0700 y snapshot/catálogo son 0600. La validación privada queda
bajo `data/`, ignorada por Git; no se publican documentos, vectores o IDs
originales en la evidencia versionada. La copia real tarda 0,21 s.

La [validación del ensayo](../research/2026-10-27-h3-legacy-quarantine-copy-validation.json)
del commit `b8b3cdd` registra **5099 passed, 1 xfailed**, sin fallos, en 212,56 s.
Incluye 26 casos nuevos, 76 pruebas dirigidas y 26 casos sobre el commit exacto.
Pasan Ruff, Bandit incluyendo el CLI sin hallazgos medios/altos, 18 tests de
frontend y el smoke del dashboard. La evidencia conserva un error del verificador
privado que intentó abrir una transacción dentro de otra; se corrigió sólo ese
verificador y se reutilizó la misma copia, sin volver a leer el origen.

El artefacto sin fuentes verifica 319 hashes y ejecuta copia y reanudación por
el CLI compilado sobre Chroma/SQLite reales y texto sintético. Dos arranques con
un índice contaminado siguen sin devolver las cuarentenas; retirar una fila
impide su recreación por retry. El ejercicio dura 0,93 s. No incluye datos
privados del hogar ni configura modelos o GPUs.

El heartbeat registra p95 de 0,18 ms y máximo de 1,03 ms; la cancelación de
control llega a 0,07 ms y el cleanup a 10,94 ms, con los presupuestos anteriores.
La aceptación integrada de GC y hardware sigue abierta.

Este paso no publica ni migra producción, no copia los índices HNSW del origen
y no completa un conjunto de rollback compatible de H6. Siguen pendientes la
revisión manual, el corte real, las demás migraciones y el borrado global de
snapshots/cuarentenas y otros colectores. El catálogo ensayado continúa privado
y separado de la versión instalada.

### Avance: journal de archivos y scopes nativos de colectores

`main` conecta ahora el driver de `COLLECTORS` a la autoridad y al coordinador
existentes. Recorre por páginas los destinos registrados de wake, correcciones
STT, conversaciones y exports; borra los archivos antes de retirar sus locators.
El I/O físico no mantiene abierta la transacción de autoridad: otra conversación
puede revocarse aunque un `unlink` esté bloqueado. Un fallo conserva el locator
para reintentar. Los exports usan un directorio propio configurado en
`training.exports_dir`.

Un recibo espera la salida real de lectores y escritores de esa conversación,
incluidos jobs todavía sin archivo reservado, feedback con sus fuentes originales
y derivados de hechos retirados. Cancelar al observador no libera esa propiedad
ni el lease. Los lectores registran su afinidad antes de adquirir texto y la
conservan hasta retirar los locales y excepciones nativos. Los recibos y el estado
de recursos exponen sólo contadores y errores fijos.

Cada arranque con lease reabre las comprobaciones de colectores, incluso para
recibos antiguos completos; un cursor restaurado no oculta datos anteriores.
Los directorios históricos siguen registrados aunque cambie la configuración.
Se auditan con descriptores, sin seguir enlaces; archivos huérfanos, legacy o
desconocidos se conservan pendientes. Se admiten raíces compartidas entre
colectores registrados y cada destino mantiene su propio recibo.

El ensayo de cuarentena registra su destino en la misma transacción de importación.
Los catálogos anteriores sin locator quedan pendientes hasta una reanudación
explícita y verificada. Las filas sin procedencia y su snapshot no se borran
automáticamente: el recibo sigue pendiente mientras cualquiera exista. Este
driver no completa los demás participantes ni habilita todavía el borrado global
en la UI. No se modifica la copia privada real ni producción.

La [validación](../research/2026-10-27-h3-collector-journal-validation.json)
del commit `780f212` registra **5137 passed, 1 xfailed**, sin fallos, en 220 s,
incluidos 38 casos nuevos. Pasan Ruff, Bandit sin hallazgos medios/altos, los
18 tests de frontend y el smoke del dashboard. Los casos nuevos cubren también
fallos de `scandir` con la excepción retenida: no quedan descriptores abiertos.
La evidencia conserva la reproducción del bloqueo de autoridad en el borrado
físico y distingue los errores de fixtures de las regresiones alcanzadas.

El artefacto sin fuentes verifica 320 hashes y ejercita los cuatro datasets con
SQLite y archivos reales, pero fuentes y audio sintéticos. Conserva tres
registros de otra conversación, espera al lector cancelado y vuelve a borrar
un archivo/locator restaurado detrás de un recibo completo. Un archivo legacy
desconocido permanece pendiente e intacto. El ejercicio tarda 0,54 s; no arranca
`main` completo ni el poller de retención, para aislar la prueba del journal.
El primer verificador esperaba siete filas e ignoraba la versión de turno
sustituida por feedback; se corrigió su cuenta a ocho sin cambiar las fuentes.

El heartbeat de la suite registra p95 de 0,22 ms y máximo de 1,06 ms; cancelar
el control tarda como máximo 0,20 ms y limpiar stream/cliente 11,21 ms.
Los presupuestos no cambian y no se manipula GC. La pausa instrumentada anterior
de 495,68 ms sigue pendiente de aceptación integrada en H4/H6.

### Avance: impedir que una rotación renombre audio atrasado

La revisión previa al driver de `PRODUCERS` encontró una vía de reintroducción:
después de rotar por 120 s de inactividad o una hora de duración, el asignador
conservaba sólo la conversación actual y permitía crear otra con un timestamp
anterior. Borrar la conversación anterior no impedía ese tercer ID. Una
reproducción aislada del código de `9e37513`, con autoridad SQLite real y datos
sintéticos, confirmó que la nueva fuente seguía autorizada aunque la original
estuviera revocada.

El inicio del intercambio actual ahora limita también la admisión y el retiro
de buffers sin fuente. Un paquete anterior no puede adquirir otro ID; un paquete
atrasado dentro del intercambio vigente conserva el ID y no hace retroceder
la última fecha de actividad. La procedencia explícita conserva su ID original
y su validez sigue dependiendo de la autoridad. El límite afecta sólo a la
habitación y origen correspondientes, sin una lista creciente de conversaciones
históricas ni I/O en el callback.

Se prueba la rotación por inactividad y duración en voz, ambient e interacción,
con y sin borrado previo. Las colas reales de comandos y tap descartan los
paquetes anteriores y conservan otra habitación. Un VAD bloqueado retiene el
array hasta su salida real, sin entregarlo a STT ni persistirlo después de la
rotación. Esto corrige una condición necesaria para el journal; todavía no
conecta ni declara completo `PRODUCERS`.

La integración pendiente debe seguir también los handoffs completos: el worker
ambiental conserva `items` y `seg` alrededor de sus awaits, mientras destilación
obtiene su procedencia después de leer el lote. Contar sólo las tasks de modelos
no demuestra que esos locales, callbacks de captura y lecturas hayan salido.

La [validación](../research/2026-10-27-h3-capture-scope-rotation-validation.json)
del commit `864c2ed` registra **5160 passed, 1 xfailed**, sin fallos, en 216,68 s,
incluidos 23 casos nuevos. La reproducción anterior alcanzó 15 fallos y tres
casos correctos; la prueba aislada del código anterior confirmó además el tercer
ID autorizado después del borrado. Pasan las 143 pruebas dirigidas finales,
Ruff, Bandit sin hallazgos medios/altos, 18 tests de frontend y el smoke del dashboard.

El artefacto sin fuentes verifica 320 hashes y ejecuta seis combinaciones de
origen/rotación, una cola de comandos real y controllers ambientales con VAD
bloqueado, tap y SQLite. El array se conserva durante la llamada nativa y se
libera al salir; ese audio retirado no llega a STT ni genera una fila en el
almacén. El ejercicio dura 0,51 s y usa únicamente datos y endpoints sintéticos.
No inicia hardware ni `main` completo, no modifica datos privados del hogar y
no publica producción. Los límites no se amplían ni se manipula GC; la aceptación
integrada de GC/hardware de H4/H6 continúa pendiente.

### Avance: scopes finitos de handoffs ambientales y lecturas de destilación

El worker ambiental de larga duración ya no conserva `items` o `seg` alrededor
de sus awaits. Cada ciclo drena y procesa en una task finita registrada antes de
adquirir arrays. Sus operaciones nativas heredan un recibo con metadata, sin
referencias fuertes al owner, resultados o excepciones. El recibo permanece
mientras un hijo real siga registrado, incluso después de cancelar al padre.
Los contadores públicos no incluyen texto, IDs, fechas ni nombres de habitación.

Destilación registra también las lecturas anteriores a conocer la procedencia.
El query real de aiosqlite vive en una task retenida: cancelar al observador no
cancela ese lector ni libera el lease durante su ejecución. La afinidad empieza
como desconocida y se resuelve con las fuentes de todas las filas adquiridas.
Se reduce al lote/prompt seleccionado sólo después de retirar los locales
excluidos. Los saves nativos mantienen esa afinidad después de salir el ciclo;
una conversación conocida ajena no queda bloqueada por ellos.

La capacidad se libera tras retirar los hijos reales, sin forzar GC. Perder el
owner no se considera prueba de salida. Una excepción del lector se sustituye
fuera de su handler para evitar conservar el traceback original con datos.
El cierre espera también los lectores retenidos antes de cerrar el catálogo.

Esta consulta de afinidad cubre trabajo finito; los buffers tap/VAD en reposo
requieren su prueba de retiro por separado. Falta integrar esos contratos con
captura, comandos y el driver de `PRODUCERS`: no se emiten recibos globales ni
se habilita el borrado amplio en la UI. Producción permanece sin cambios.

La [validación](../research/2026-10-27-h3-producer-scope-handoffs-validation.json)
del commit `063b444` registra **5179 passed, 1 xfailed**, sin fallos, en 223,22 s,
incluidos 19 casos nuevos y 120 pruebas dirigidas finales. Pasan Ruff, Bandit sin
hallazgos medios/altos, 18 tests de frontend y el smoke del dashboard. Los tests
incluyen una función SQLite bloqueada en el thread real, cancelación anterior
a conocer las fuentes, afinidad de saves nativos, selección de fuentes por
presupuesto/idioma, capacidad retenida y pérdida del owner sin falsa prueba de salida.

El artefacto sin fuentes verifica 321 hashes y ejecuta la lectura SQLite retenida,
el save nativo revocado y un handoff ambiental pausado después de salir del
modelo. El array se libera al salir el handoff; el controller de larga duración
no tiene locales de audio. El ejercicio dura 0,28 s, con datos y endpoints
sintéticos. El primer verificador pasaba `metadata` al ledger y fallaba antes
de su control de autoridad: su resultado se descarta como prueba del rechazo.
El verificador corregido exige la llamada real y su `PermissionError` por fuente
revocada; no modifica las fuentes de aplicación ni los límites de aceptación.

El heartbeat registra p95 de 0,33 ms y máximo de 1,09 ms; cancelar el control
tarda como máximo 0,08 ms y limpiar stream/cliente 10,96 ms. No se manipula GC
ni se afirma aceptación integrada de GC/hardware o arranque completo de `main`.

### Avance: afinidad finita de comandos y controladores de captura

El procesador de comandos conserva ahora los scopes de sus controladores,
sombras y llamadas nativas reales. La detección directa de emoción también
tiene un controlador registrado y comparte el límite de admisión de ocho
operaciones. La consulta por conversación retira primero las asociaciones
idle de usuario/emoción que perdieron su fuente; los perfiles de voces siguen
bajo su política propia. Cancelar al observador no libera la afinidad del
thread real ni convierte una conversación conocida ajena en trabajo pendiente.

Cada `CaptureRoom` registra las llamadas auxiliares y pasos antes de esperar
el lock, con ocho controladores como máximo. El cierre cancela a los que esperan
y conserva el worker que ya entró. Las llamadas auxiliares fijan la procedencia
explícita o del llamador al admitirse; no la sustituyen por otra al salir de la
cola. Se comprueba la fuente antes y después del callback, impidiendo devolver
un resultado que quedó revocado durante su ejecución.

Se reprodujeron tres fallos en el draft: perder la fuente del llamador permitía
entrar después de revocarla; una excepción retenía argumentos crudos; y el nuevo
rechazo por capacidad conservaba argumentos si fallaba la entrada al scope.
Mover el error fuera del handler nativo no bastaba: el transporte del executor
también conserva sus tuplas de argumentos en el traceback. El callback ahora
devuelve un DTO con resultado o error fijo; la excepción se crea en el loop,
sin transportar el traceback original. Los rechazos limpian también los
argumentos que nunca llegaron a admitirse.

Estos contratos cubren trabajo finito. Falta el retiro verificable de buffers
en reposo, los handoffs de los loops que reciben los resultados, callbacks de
entrada y la integración completa con el driver de `PRODUCERS`. No se habilita
el borrado amplio ni se cambia producción.

La [validación](../research/2026-10-27-h3-command-capture-scopes-validation.json)
del commit `67e589e` registra **5203 passed, 1 xfailed**, sin fallos, en 219,31 s,
incluidos 24 casos nuevos y 155 pruebas dirigidas finales. Pasan Ruff, Bandit
sin hallazgos medios/altos, 18 tests de frontend y el smoke del dashboard.
Se conservan las reproducciones de los tres fallos del draft y del intento
insuficiente de crear la excepción dentro del worker. Dos errores de fixtures
se distinguen de esos fallos alcanzados: una inserción antes de cerrar un
`try/finally` y un wake sintético que no disparaba el VAD que se esperaba probar.

El artefacto sin fuentes verifica 321 hashes y ejercita la afinidad tras cancelar
el comando, el controlador directo de emoción con lease retenido, una entrada
auxiliar revocada que nunca entra y excepciones observadas que no mantienen
argumentos crudos. La saturación sintética reduce sólo su capacidad de metadata
a uno para forzar el rechazo antes del worker. El ejercicio dura 0,49 s; no
cambia la configuración instalada ni los límites de aceptación.

El heartbeat registra p95 de 0,29 ms y máximo de 1,07 ms; cancelar el control
tarda como máximo 0,10 ms y limpiar stream/cliente 10,95 ms. Los presupuestos
no cambian y no se manipula GC. Continúan pendientes el arranque completo de
`main`, la aceptación integrada de GC/hardware y los demás hitos de H0–H6.

## Trabajo restante

1. Resolver el ciclo de vida y revisión de datasets legacy, snapshots y adapters,
   cubrir los demás consumidores de clips/aprendizaje y conectar
   todos los productores reales al journal. El procesador y ambient retenidos
   todavía no completan `PRODUCERS`.
2. Propagar procedencia y TTL a los demás logs de diagnóstico, colectores,
   streams y frontend; auditar las demás rutas de persistencia, caché y
   consumidores alternos de prompts, incluido el
   `RequestProcessor` antiguo no cableado.
3. Coordinar el borrado autenticado de conversación entre capturas, clips,
   contexto corto, resúmenes, hechos e índices; conservar tombstones y reintentos
   frente a productores y guardados tardíos. Conectar al journal los participantes
   restantes y los scopes de destilación ya retenidos; contrastar los destinos y clips
   legacy indeterminados del inventario; completar la reconstrucción del índice
   vivo y la restauración compatible con autoridad vigente. Agregar selección y recibos en el
   dashboard autenticado después de cerrar esas garantías. La limpieza local
   del catálogo y del índice no sustituye la operación completa.
4. Implementar pausa ambiental y mute total con reactivación independiente del
   audio; definir qué productores se detienen y qué buffers se descartan.
5. Aplicar 48 horas a huellas y grupos acústicos no confirmados; conservar perfiles
   confirmados manualmente hasta borrado, registrando la revisión verificable.
6. Construir agrupación y revisión de voces, correcciones de unión/separación y
   abstención. Fijar métricas y corpus de calibración antes de evaluar el resultado.
7. Completar las demás migraciones en copia y la aceptación de runtime, micrófono,
   reproducción, HA y rollback durante H6. La restauración debe conservar la
   intención de borrado y evitar reintroducir datos eliminados. No se ha
   modificado producción.
