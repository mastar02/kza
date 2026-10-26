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

## Trabajo restante

1. Propagar procedencia y TTL a los demás logs de diagnóstico, colectores,
   streams y frontend; auditar otras rutas de persistencia y caché e integrar
   sugerencias derivadas y consumidores alternos de prompts, incluido el
   `RequestProcessor` antiguo no cableado.
2. Coordinar el borrado autenticado de conversación entre capturas, clips,
   contexto corto, resúmenes, hechos e índices; conservar tombstones y reintentos
   frente a productores y guardados tardíos. Conectar al journal los participantes
   restantes y los lotes de destilación en curso; contrastar los destinos y clips
   legacy indeterminados del inventario; completar la reconstrucción del índice
   vivo y la restauración compatible con autoridad vigente. Agregar selección y recibos en el
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
