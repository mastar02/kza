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

`build_memory_services` construye y registra el cierre del manager antes de
inicializarlo, y mantiene un único worker de vencimiento y reconciliación.
Los workers nativos son acotados y siguen registrados si se cancela al llamador.
El cierre informa recursos retenidos si no terminan dentro de su presupuesto;
no presenta un thread todavía activo como terminado. Una lease de archivo impide
dos escritores del mismo catálogo persistente.

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

## Trabajo restante

1. Propagar procedencia y TTL a los demás logs de diagnóstico, colectores,
   streams y frontend; auditar otras rutas de persistencia y caché e integrar
   sugerencias derivadas y consumidores alternos de prompts, incluido el
   `RequestProcessor` antiguo no cableado.
2. Coordinar el borrado autenticado de conversación entre capturas, clips,
   contexto corto, resúmenes, hechos e índices; conservar tombstones y reintentos
   frente a productores y guardados tardíos. La limpieza local del catálogo no
   sustituye esa operación.
3. Implementar pausa ambiental y mute total con reactivación independiente del
   audio; definir qué productores se detienen y qué buffers se descartan.
4. Aplicar 48 horas a huellas y grupos acústicos no confirmados; conservar perfiles
   confirmados manualmente hasta borrado, registrando la revisión verificable.
5. Construir agrupación y revisión de voces, correcciones de unión/separación y
   abstención. Fijar métricas y corpus de calibración antes de evaluar el resultado.
6. Probar migración en copia y completar aceptación de runtime, micrófono,
   reproducción, HA y rollback durante H6. No se ha modificado producción.
