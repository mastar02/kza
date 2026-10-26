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
agrega un endpoint global ni declara completo el borrado de conversación. Resta
mantener la purga de 48 h sobre todos los destinos registrados cuando ambient
está apagado y sobre raíces anteriores sin un pedido de borrado; la purga actual
del transcriber continúa dependiendo del ambient path activo. La migración y
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

## Trabajo restante

1. Propagar procedencia y TTL a los demás logs de diagnóstico, colectores,
   streams y frontend; auditar otras rutas de persistencia y caché e integrar
   sugerencias derivadas y consumidores alternos de prompts, incluido el
   `RequestProcessor` antiguo no cableado.
2. Coordinar el borrado autenticado de conversación entre capturas, clips,
   contexto corto, resúmenes, hechos e índices; conservar tombstones y reintentos
   frente a productores y guardados tardíos. Conectar al journal los participantes
   restantes y los lotes de destilación en curso; mantener la purga de capturas
   históricas aunque ambient esté apagado; ampliar la limpieza sin recall
   a vencimientos y borrados individuales. Agregar selección y recibos en el
   dashboard autenticado después de cerrar esas garantías. La limpieza local
   del catálogo y del índice no sustituye la operación completa.
3. Implementar pausa ambiental y mute total con reactivación independiente del
   audio; definir qué productores se detienen y qué buffers se descartan.
4. Aplicar 48 horas a huellas y grupos acústicos no confirmados; conservar perfiles
   confirmados manualmente hasta borrado, registrando la revisión verificable.
5. Construir agrupación y revisión de voces, correcciones de unión/separación y
   abstención. Fijar métricas y corpus de calibración antes de evaluar el resultado.
6. Probar migración en copia y completar aceptación de runtime, micrófono,
   reproducción, HA y rollback durante H6. La restauración debe conservar la
   intención de borrado y evitar reintroducir datos eliminados. No se ha
   modificado producción.
