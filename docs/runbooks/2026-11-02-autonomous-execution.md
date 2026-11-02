# Continuidad de la ejecución autónoma

El objetivo activo ejecuta el perfil de
[roadmap](../plans/2026-11-02-autonomous-roadmap.md). El único escritor de código
y Git es la sesión raíz; los subagentes investigan y revisan sin editar.
Se conserva la rama existente `recovery/production-2026-10-25`, bajo `yo` y en
`/home/yo/projects/kza`. No se crean servicios ni automatizaciones de reanudación.

## Contrato congelado

`python3 tools/roadmap_status.py` reproduce los hashes originales con SHA-256
sobre JSON UTF-8: `json.dumps(value, sort_keys=True, ensure_ascii=False)` con
separadores predeterminados de Python. Se preserva el orden de los arrays.

El digest de metas usa `criteria` sin `status`/`evidence`, más `stratification`.
El digest de perfil usa `targets` (el digest anterior), `decisions`
(`confirmed_decisions`) e `integrations` (`integration_portfolio`). Los valores
congelados son los del commit de activación `162db16`; no se recalculan para
silenciar discrepancias. Cambios del contrato requieren decisión registrada
antes de una nueva reserva. Actualizar estados no cambia las metas.

El verificador revisa referencias, evidencia mínima de estados y dependencias.
Es una comprobación estructural: una ruta de evidencia no prueba su contenido
ni su cobertura. El revisor debe abrir la evidencia actual y comprobar el alcance.

## Checkpoint privado y propiedad

`data/autonomous-roadmap-20261102/checkpoint.json` conserva el ID real de sesión
de Codex, ID de hilo, usuario, checkout, rama, revisión, hash del manifiesto,
última unidad, próximo trabajo y handles de procesos pendientes. El directorio
tiene modo 0700 y los archivos de evidencia 0600. No se publica el ID de sesión.

Al retomar, confirmar identidad y Git; leer manifiesto, checkpoint y evidencia
de la última unidad. Confirmar que no hay otro escritor activo antes de asumir
propiedad. Un checkpoint o lock por sí solo no demuestra que un proceso siga
vivo. Consultar el handle real y su salida antes de repetir trabajo; registrar
si terminó o si su handle dejó de existir. Usar `codex exec resume` sólo con el
ID privado comprobado de esta ejecución. No usar selección de última sesión.

## Registro de evidencia por unidad

Cada registro privado contiene `task_ids`, `source_revision`, `worktree_patch_sha256`
si hay cambios aún no commiteados, `command`, `started_at_utc`, `finished_at_utc`,
`exit_code`, `environment` y `artifacts` con ruta y SHA-256. El comando queda sin
secretos. Se distingue fixture/local, observación real de lectura y campo
autorizado. La evidencia sensible sólo se referencia desde el manifiesto público.

Comprobar resultados, actualizar tarea/fase y guardar commit/push por unidad.
Si el SHA de implementación se conoce después de crear el commit, registrarlo
en el siguiente checkpoint; evitar referencias circulares a un SHA propio.
Conservar errores y primeros resultados; las repeticiones no los sobrescriben.
No promover un requisito compartido a `accepted` al verificar una preparación.

## Límites de esta preparación

La reconciliación HA usa únicamente archivos privados ya recibidos. Se comparan
hashes, selecciones, operaciones, rangos y evidencia pendiente sin cambiar
aprobaciones ni activar `home_assistant.control_selection_path`. La selección
histórica no prueba disponibilidad ni efectos en vivo. Esa verificación y las
fuentes reales se mantienen para sus tareas posteriores.

El [contrato del corpus](../plans/2026-11-02-evaluation-corpus.md) mantiene
separados datos ordinarios y evaluación consentida. P0 prepara política y
fixtures; no permite atribuir resultados humanos, acústicos o físicos.

## Grafo efectivo y laboratorio

`src.main.build_application(config, shutdown=stack, factories=...)` es la misma
fábrica que utiliza `main()`. Devuelve `ApplicationGraph` con pipeline,
Casa/HA, privacidad, managers, servicios de voz/web y modelos compartidos.
Las dependencias hoja omitidas se resuelven al llamar la fábrica; los tests no
reconstruyen políticas o managers por su cuenta. La fábrica puede inicializar
almacenamiento y mantenimiento privado: no es una función pura.

El llamador conserva un `AsyncExitStack` y lo cierra incluso si la construcción
falla o se cancela. Después de construir, `graph.warmup()` precalienta las mismas
instancias; `graph.run()` inicia sincronización, rutinas, presencia, recordatorios,
listeners y voz. No hay un modo de construcción alternativo para tests. Warmup
de STT/SpeakerID/Emotion usa el dueño nativo del procesador; TTS usa el dueño de
respuestas. Las cancelaciones no liberan privacidad/HA antes del trabajo real.

El cierre operativo retiene su task ante cancelaciones repetidas, observa los
handles de dashboard/ambient y ejecuta todos los pasos aunque falle un stop.
Detiene rutinas antes de retirar la pipeline; el stack externo conserva los
callbacks como respaldo para fallos anteriores al arranque. El cierre completo
del thread legacy de entrenamiento sigue sujeto al inventario de T022.

`tests/integration/test_application_graph.py` usa esta fábrica con referencias
sintéticas, cliente HA real contra loopback, Chroma/SQLite reales y adaptadores
de modelos/audio. Incluye cuatro entradas configuradas sin abrir dispositivos,
identidad de dependencias, frase → timer → PCM, selección vacía y cancelaciones
de construcción/warmup/ejecución. Las fuentes y salidas simuladas se identifican
como tales. Las rutas privadas quedan en el directorio temporal, incluida la
de métricas; los listeners se sustituyen para no ocupar puertos del servidor.
Estas pruebas no sustituyen aceptación física ni completan las familias de P3.

## Recibos de operación y respuesta

`TaskOutcome` registra estado de la tarea, persistencia, envío, efecto y alcance
de observación. `DeliveryOutcome` registra generación y entrega real de voz sin
texto ni excepción privada. `TurnOutcome` los compone por solicitud: `handled`
sólo indica reconocimiento; `task_completed` exige evidencia de la operación;
`response_completed` exige la entrega requerida; `turn_completed` y `success`
requieren ambas. Los éxitos domésticos silenciosos tienen respuesta `silent`
y `response_completed=null`, sin fabricar una entrega audible.

Una aceptación HA no demuestra efecto. Los recibos conservan cada hoja de una
operación compuesta y sólo declaran `observed` con alcance `ha_state` tras leer
el estado correspondiente. Esto tampoco demuestra físicamente la luz o motor.
El ID interno del turno llega a ambos routers y al servicio compartido: una
cancelación del llamador no autoriza otro envío para el mismo ID. Los recibos HA
siguen siendo volátiles; T050/T051 cubren durabilidad y autoridad posterior.

Listas y recordatorios acreditan commits locales; eliminar un objeto ausente o
una fila que desapareció tras buscarla no completa la operación. Vaciar una
lista verifica que exista y esté vacía después del commit. La réplica opcional
de listas hacia HA queda en otro campo: una réplica fallida no borra ni repite
la escritura local. No establece todavía la autoridad entre HA y KZA de T051.
Los timers actuales declaran persistencia `volatile`, anuncios en espera quedan
`queued` y las creaciones de recordatorio sin parser temporal quedan
`unsupported`; esas familias se completan en P3. Las rutas legacy sin prueba
explícita conservan `unknown` aunque hayan respondido una frase.

Voz, métricas de disco/TCP y eventos live transmiten estados separados. Las
proyecciones diagnósticas permiten únicamente booleanos y códigos cerrados;
omiten recibos, IDs, texto y motivos libres. `test_from_text()` es diagnóstico
de reconocimiento: no acredita ejecución ni entrega. Las pruebas con la
fábrica efectiva cubren fallos de transporte, PCM, réplica, objetos ausentes,
efectos parciales y cancelación concurrente. La aceptación física sigue pendiente.
