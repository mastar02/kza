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

## Observación, capacidad y caducidad

`Observation` conserva fuente cerrada, estado, instante observado, vencimiento y
último intento por separado. Un intento fallido, una respuesta HTTP o un ACK de
acción no renuevan la observación. Una fecha ausente o futura no permite declarar
vigencia. Las observaciones parciales identifican los campos no observados.

`HomeCapabilityService` pertenece al grafo efectivo y comparte una lectura
simultánea entre voz y Casa, con políticas explícitas y contexto de tarea vacío.
No cachea decisiones de selección entre consultas. Las capacidades efectivas
vencen con la primera dependencia que caduque: estados, selección y configuración
de temperatura cuando corresponda. Primero se resuelve un destino entre todos
los grants actuales; su falta de observación nunca permite elegir otro.

El servicio conserva el trabajo ante cancelaciones del llamador y drena sus
hijos antes de retirar HA. El cliente serializa también las lecturas que inicia
su sync interno. Los consumidores vuelven a consultar el reloj después de una
espera; Casa ancla la vigencia a tiempo monotónico e incluye los vencimientos
individuales de las entidades. La acción sigue pasando por la revalidación final
de `HomeActionService`; el descubrimiento no concede permisos.

Las colecciones técnicas añaden `X-KZA-Source-Id`, estado de observación,
`Observed-At`, `Expires-At` y campos no observados. `Expiry-Scope=observation`
describe frescura, sin extender retención de contenido. Una lista real vacía
conserva fuente real; un fallo conserva las fechas anteriores y declara
indisponibilidad. El cache guarda sólo metadata de nueve componentes, sin
payloads ni filtros personales, y no retrocede ante respuestas antiguas.
La demo requiere configuración explícita; los valores inferidos quedan
desconocidos o parciales, sin acreditar salud, muestras o permisos efectivos.

Los probes nativos comparten trabajo acotado fuera del event loop y fijan su
observación una sola vez al terminar. El cierre intenta todos sus dueños incluso
si falla el servidor o el catálogo. Los headers de conversaciones se construyen
tras la consulta real, dentro del dueño ASGI; su frescura se limita además por
el vencimiento de las filas. Los receipts y las guardas posteriores a headers
permanecen como autoridad de entrega y revocación.

El clima actual del orquestador utiliza la misma lectura tipada y la entidad
configurada exacta, sin volver al cache sin fecha. El pronóstico conserva su
consulta acotada y distingue lectura vacía, inválida y fallida. Su fecha de
consulta no acredita emisión ni alineación de fechas del proveedor; el formatter
posicional heredado sigue pendiente de comprobación en T037/T060. La lectura de
estado HA tampoco acredita el instante de medición meteorológica física.

Esta infraestructura no acepta C06 por sí sola. La procedencia de cada endpoint,
la presentación completa de desconocidos y la renovación de las vistas mantienen
sus verificaciones propias de T012, T040 y T046.

T012 queda verificado localmente: la suite completa sobre `9ccb288` registra
5925 casos aprobados y el único `xfail` previo. El ajuste final `b56b32a` exige
demo explícita también en la fábrica y rechaza limpiar un cooldown sin su
dependencia; sus 112 pruebas focalizadas pasan sobre esa revisión comprometida.
La evidencia pública conserva ambas revisiones, hashes y primeros intentos en
`docs/research/2026-11-02-autonomous-t012-validation.json`. No acredita criterios
de aceptación ni sustituye las comprobaciones pendientes de clima en T037/T060.

## Autoridad de sesiones en preparación

La base de T013 define roles cerrados de hogar, operación y evaluación temporal,
concesiones persistidas sólo como hashes y cookies individuales con vencimiento
y revocación. Una retirada sella admisión antes de SQLite; una concesión se
publica después del commit. La revisión conserva el secreto pero invalida las
sesiones y capacidades anteriores. La rotación administrativa conserva épocas
para no reactivar cookies antiguas al volver a un valor previo.

El dueño conserva transacción y publicación ante cancelaciones, y mantiene el
lease de escritor hasta terminar el trabajo nativo real. Persiste también los
vencimientos de muestras sin renovar sus fuentes ni ampliar modalidades.
La revisión `89f3db9` pasa 102 pruebas focalizadas; su evidencia privada está en
`data/autonomous-roadmap-20261102/t013/foundation-evidence.json`.

La fábrica comparte esa autoridad entre `ApplicationGraph`, `DashboardAPI` y
`app.state`. Registra el cierre antes de los consumidores e inicializa el
almacén antes de los canales; el dashboard no cierra la dependencia inyectada.
La revisión `3953b1e` pasa 128 pruebas focalizadas de grafo, configuración y
compatibilidad. El laboratorio aísla tanto su base como credenciales heredadas.

La revisión `ce63071` conecta login/logout y gates basales por ruta. Las cookies
son opacas; cada petición y operación retenida conserva su lease original.
Una autoridad inyectada no disponible deniega acceso sin recurrir al token
administrativo. La integración standalone sin autoridad conserva sólo ese
bearer, sin cookies compartidas ni emisión de sesiones duraderas.

La unidad HTTP pasa 824 pruebas sobre la revisión comprometida, incluidos
controles HA de laboratorio con bearer y cookie, y retirada de sesión durante
inscripción vocal nativa. El cierre conserva épocas y suelo temporal observados
sin una nueva petición. Los detalles permanecen en
`data/autonomous-roadmap-20261102/t013/http-evidence.json`.

La revisión `6141abb` conserva el lease original en controles domésticos tras
esperar lanes, preflight y el lock de sesión de HA. Comprueba otra vez justo
antes de POST y distingue `NOT_SENT` de rechazo remoto e incertidumbre. Cada
hoja mantiene su autoridad: una retirada posterior a una hoja observada deja
un recibo parcial y detiene las restantes. Reunirse al mismo ID desde otra
sesión no sustituye la guarda original ni reenvía el efecto.

Esta unidad pasa 978 pruebas sobre la revisión comprometida, incluidas 17 de
autoridad con fábrica/transporte reales de laboratorio. La validación de debug
también corrigió fixtures que dejaban abierto un canal de llamadas a un host
dummy; los clientes quedan ahora con dueño y cierre completos. La evidencia
privada está en `data/autonomous-roadmap-20261102/t013/native-evidence.json`.

La revisión `82388d7` conserva la autoridad original durante confirmación,
rechazo, borrado y retirada de conversaciones en memoria. La comprueba después
del lock y antes del commit nativo; una retirada revierte sólo esa transacción.
El dueño completa la reconciliación de una escritura ya comprometida aunque
se cancele el observador o termine su sesión. La evidencia privada está en
`data/autonomous-roadmap-20261102/t013/memory-evidence.json`.

T013 sigue en curso: faltan guards de otras escrituras asíncronas, prevención de
lecturas personales desde voz anónima, ámbitos personales y de evaluación,
binding HTTP/WS, guardas de todos
los cuerpos privados, gestión de credenciales y navegación por rol. No acredita
accesos reales ni criterios de aceptación.
