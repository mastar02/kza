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
rechaza. Los snapshots nuevos privados viven bajo `contexts/policy_v2`, con
procedencia explícita. Los JSON legacy permanecen fuera de esta lectura, sin
reescribirlos. El worker elimina snapshots conocidos vencidos o revocados y
temporales que no se promovieron, también al arrancar.

La compacción comprueba identidad y revisión del contexto antes de publicar,
y vuelve a validar sus fuentes tras esperar al reasoner. Limpiar o reemplazar
un contexto invalida su resultado tardío. El snapshot final conserva un contexto
que volvió a activarse mientras esperaba. La carga y escritura de disco se
ejecutan fuera del loop, sin mantener el lock del contexto durante I/O. Se
registran los trabajos nativos incluso si se cancela al llamador; el cierre
espera su salida real o comunica el vencimiento de su presupuesto.

Esta integración no expone todavía un borrado amplio de conversación: faltan
las capturas/clips y otras copias, sus productores y la coordinación durable
desde el dashboard. Tampoco transforma los perfiles de usuario ni las
preferencias legacy en confirmaciones manuales.

## Trabajo restante

1. Propagar procedencia y TTL a eventos y colectores que aún conservan copias
   de texto o audio, y auditar otras rutas de persistencia y caché.
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
