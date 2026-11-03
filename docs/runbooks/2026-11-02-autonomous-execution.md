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

Las revisiones `79a78ab` y `72d77c5` comprueban el lease original antes de headers
y de cada fragmento privado exterior. Una retirada anterior produce un 403
completo sin esperar al generador original; después de headers interrumpe la
respuesta sin sustituir su cuerpo ni declarar éxito. Login comprueba también
la sesión recién emitida antes de publicar su cookie. Los efectos comprometidos
y tickets pendientes mantienen su dueño. Esta unidad pasa 626 pruebas;
`data/autonomous-roadmap-20261102/t013/body-evidence.json` conserva el detalle.

### Persistencia y publicación de tareas

La revisión `4289883` reúne cada transacción y lectura de listas en un callback
nativo de SQLite, con guarda original antes del commit y rollback propio.
Los tres mutadores HTTP propagan esa guarda; borrar un ítem comprueba también
la lista de la ruta. Un fallo de rollback bloquea proyecciones posteriores.
El cierre conserva su dueño y conexión hasta terminar el trabajo real; el lock
de ciclo de vida impide sustituirla mientras drena. La unidad pasa 797 pruebas;
`data/autonomous-roadmap-20261102/t013/sqlite-list-evidence.json` contiene límites
y evidencia. Los ámbitos personales siguen pendientes.

La revisión `63334d2` aplica el mismo dueño a todas las operaciones de
recordatorios. La sesión original llega hasta el commit y la tarea retenida
publica después el aviso al scheduler. Un evento compartido evita referencias
circulares; limpiarlo antes de consultar conserva el aviso que llega durante
la consulta vacía. Un fallo del aviso conserva el recordatorio escrito sin
acreditar una entrega futura. La unidad pasa 883 pruebas, con evidencia en
`data/autonomous-roadmap-20261102/t013/reminder-evidence.json`.

La revisión `56be9af` aplica la transacción nativa a rutinas y conserva claim y
publicación del worker dentro del mismo dueño. El drenaje no cerrante incluye
esa publicación antes de interrumpir workers y permite guardar sus recibos.
Una generación de stop evita que un arranque concurrente vuelva a publicar
estado activo. Las seis mutaciones HTTP mantienen su guarda original; los
trabajos aprobados usan su snapshot y la política doméstica vigente. La unidad
pasa 1094 pruebas; el detalle está en
`data/autonomous-roadmap-20261102/t013/routine-evidence.json`.

### Contexto de voz anónima

Las revisiones `c37f6d5`, `0bfccb6` y `83eb465` separan el contexto doméstico
efímero de los contextos personales. La voz no recupera recuerdos, preferencias,
perfiles ni resúmenes personales, tampoco bajo `unknown`. La prueba con
`build_application` en ambos modos prepara datos privados recuperables y
snapshots válidos, comprueba su ausencia del prompt y la respuesta y conserva
la procedencia de la captura actual. Los contextos domésticos no guardan
historial libre ni cargan, compactan o escriben snapshots.

La cola separa propietarios domésticos por una clave interna de la habitación
original y propietarios backend por usuario. Cancelación y reemplazo usan la
misma separación, independientemente de la zona de salida. Las proyecciones
públicas revisadas omiten las claves domésticas. Retirar una fuente privada no
consultada conserva el turno independiente; retirar la captura actual cancela
generación, entrega y procesamiento posterior. Los positivos de selección
backend explícita conservan las pruebas nativas de transferencia y cierre.

La revisión `0174104` pasa la suite completa: 6147 pruebas y una exclusión
conocida. Un diagnóstico vinculó la pausa de las corridas previas a la
recolección de 215456 objetos acumulados por otros fixtures. La prueba retira
ese estado antes de medir, mantiene GC y umbrales activos y cierra cada dueño
de respuesta. En diez repeticiones mide una pausa máxima de 1,09 ms y una
cancelación máxima de 1,69 ms. Esta medición sintética no acredita el peor caso
del proceso completo bajo deuda de GC o carga física. La unidad queda
verificada localmente; la evidencia y todos los intentos están en
`data/autonomous-roadmap-20261102/t013/household-evidence.json`.

### Listas compartidas por voz

La revisión `db69370` filtra listas compartidas en SQLite antes de seleccionar
homónimos, construir DTOs o buscar artículos. Cada escritura comprueba de nuevo
el padre dentro de su transacción, antes de tocar hijos. La voz no autocrea una
lista personal bajo `unknown`; crear requiere pedir una lista compartida, sin
atribuir autoría a un perfil acústico. Una escritura rechazada no replica a HA.

Los GET privados requieren administración y llevan la guarda original hasta
la proyección nativa. Retirar la sesión mientras la lectura espera impide leer
los campos privados, incluso si se canceló su observador. Añadir a un ID
inexistente devuelve 404 sin acreditar persistencia. Los casos positivos de
persistencia y reinicio preparan explícitamente listas compartidas.

La unidad pasa 1134 pruebas, con grafos reales en ambos modos y salida por la
habitación configurada. Revisión y Ruff no encuentran bloqueantes. El detalle
y los intentos están en
`data/autonomous-roadmap-20261102/t013/shared-list-evidence.json`.

### Recordatorios compartidos y ocurrencias

Las revisiones `5fc095a` y `baeeae5` añaden un ámbito cerrado privado/compartido.
Los registros anteriores, incluidos los de `unknown`, y las creaciones sin
declaración explícita siguen privados. Voz, scheduler, llegadas y acciones
filtran compartidos en SQL; releen el ID antes de usar su contenido. Los GET
privados conservan la guarda administrativa original; el estado del scheduler
cuenta sólo trabajos compartidos mediante el mismo dueño de SQLite.

La relectura conserva la ocurrencia seleccionada y exige que siga activa y
vencida. Cada transición compara también fecha y estado dentro de su UPDATE.
Una llegada retenida mientras otra entrega avanza a la siguiente ocurrencia
no ejecuta esa sucesora ni sobrescribe su fecha, incluso si también está
vencida. Repetir una ocurrencia antigua, futura, cancelada o completada no
duplica efectos ni transiciones. Se conserva el aviso publicado después del
commit y su prueba de creación durante una consulta vacía.

La unidad pasa 1230 pruebas y Ruff. Los grafos reales en ambos modos conservan
privados activos y silenciosos y entregan el compartido con un único efecto
HA sintético. La revisión encontró la carrera de recurrencia, se corrigió y
quedó cubierta por pruebas deterministas. El detalle está en
`data/autonomous-roadmap-20261102/t013/shared-reminder-evidence.json`.

T013 sigue en curso: faltan guards de mutaciones administrativas nativas,
ámbitos personales y
de evaluación, binding HTTP/WS, gestión de credenciales y navegación por rol.
No acredita accesos reales ni criterios de aceptación.

### Administración nativa y catálogo canónico

El dashboard conserva un único dueño acotado para trabajo administrativo nativo,
con contexto vacío y la guarda de la sesión original. El catálogo, los reinicios
y el enrollment legacy se ejecutan fuera del loop. Cancelar el observador no
cancela el trabajo ya admitido: el cierre sella la admisión y espera su salida,
el servidor y los recursos. El pipeline recibe ese drenaje como dependencia
antes de cerrar sus componentes. Un fallo de limpieza puede reintentarse sin
reabrir la admisión ni repetir una operación administrativa.

UserManager protege un catálogo separado de sus copias públicas. Cada mutación
valida la cuenta solicitante actual, prepara una copia, escribe y sincroniza un
temporal privado y vuelve a comprobar autoridad antes de reemplazar el archivo.
Sólo después publica RAM y versión; una revocación posterior no deshace el
commit aprobado. Los IDs nuevos son UUID y no reutilizan identidades borradas.
Un catálogo inválido no publica filas parciales ni simula una lectura vacía.
Las cuentas canónicas se cargan también con Speaker ID deshabilitado, sin
importar embeddings legacy ni construir modelos biométricos.

Enrollment usa el mismo bloqueo nativo del modelo que el procesador de voz;
limita metadatos y frames antes de decodificar y rechaza arrays vacíos o no
finitos después de convertir. Publica únicamente el resultado pequeño de la
operación. Reinicio distingue resultado del comando y efecto incierto, sin
reintentos automáticos. Cooldown publica tras el reemplazo; alertas comprueba
autoridad dentro de su bloqueo y conserva las reglas de timestamp de ack/dismiss.
Las lecturas de cuentas, cooldown y métricas revalidan al adquirir sus bloqueos.

Las revisiones `ade7a98`, `adab7ec` y `69ea99c` pasan 995 pruebas cruzadas con
componentes reales y entradas sintéticas. La suite completa de `69ea99c` pasa
6229 pruebas y conserva una xfail conocida; no se suman ambos conjuntos.
Ruff no informa errores y los hashes de las fuentes no cambiaron durante la
ejecución. La evidencia y los intentos anteriores se conservan en
`data/autonomous-roadmap-20261102/t013/admin-evidence.json`.
T013 conserva pendientes los ámbitos personales y de evaluación, binding
HTTP/WS, recibos, credenciales y navegación por rol. Por instrucción del usuario,
la siguiente unidad será T047: integrar la referencia NEXO aprobada en el
frontend con backend real y mantener separado el vínculo vocal de la autoridad.

## Referencia NEXO y frontend real

T047 forma parte del goal activo por instrucción del usuario del 2026-11-02.
La referencia aprobada está en
`/home/yo/projects/kza-design-reference/nexo-2026-11-02-d686fcd`.
Se leyeron `LEEME.txt`, `HANDOFF.json` y las capturas y se verificó
`MANIFEST.sha256`. Los estilos, la geometría amueblada, Three.js e Inter
proceden de ese paquete; los recursos y sus licencias se sirven localmente.

### Entrada, Hogar y Acciones

La primera unidad conserva el tema oscuro, cian/violeta, tipografía, navegación
adaptable y modelo 3D. Los ambientes y controles proceden del catálogo real,
con sus IDs, capacidades y límites. Luces, clima y persianas conservan órdenes
independientes y recibos; un POST perdido se reconcilia por GET con su mismo
identificador, sin repetir el envío. La temperatura se identifica como objetivo,
no como temperatura ambiente. Los campos opcionales no cambian otros ajustes
sin elección explícita.

La casa permanece ilustrativa: no hay un plano real suministrado ni se infiere
una relación entre habitaciones del modelo y dispositivos por sus nombres.
El renderer cancela su RAF, observador y listeners y dispone recursos al salir
de la vista o de la sesión. La navegación de cámara conserva estado visual y ARIA.

La nueva entrada consulta quién es la sesión y sus permisos antes de acceder
al catálogo. No obtiene autoridad de voces o nombres. Retira datos, controles
y renderer al vencer o cerrar sesión; serializa login/logout y no recupera
datos tras un cierre fallido ni desde bfcache sin volver a comprobar autoridad.
Los relojes consideran suspensión y retroceso civil sin renovar vencimientos.

La unidad pasa 91 pruebas de JavaScript, 715 pruebas de rutas/autoridad/grafo
y Chromium con login HTTP individual de Hogar, Evaluación y Administración.
El navegador verifica efectos HA sintéticos, recibos pendientes, aislamiento,
cancelación de recursos, foco/borradores y reflow a 320 y 390 píxeles.
Las capturas contienen exclusivamente datos de prueba; no acreditan dispositivos
reales, aceptación física ni el cierre completo de T047 o T013.
Las revisiones son `0c1fa93` y `f80bfe2`; la evidencia, capturas y los intentos
anteriores están en
`data/autonomous-roadmap-20261102/t047/home-shell-evidence.json`.

Voces todavía abre la vista técnica existente mediante un destino permitido.
Las herramientas anteriores se conservan en `/home/legacy.html`.
Las siguientes unidades adaptarán Voces a NEXO y añadirán el contrato
manual perfil vocal–persona: metadata administrativa sin crear sesión,
credencial, consentimiento, roles o acceso personal. T047 sigue en curso.

### Catálogo de Personas y retiro de cuentas

La base para Personas incorpora `GET /api/admin/users`, limitado a ID, nombre,
nivel de cuenta y estado activo. La proyección se realiza dentro del lock con
la autoridad original; no copia vectores, preferencias ni el objeto completo.
Un catálogo no verificable devuelve indisponibilidad, no una lista vacía.
El nivel de cuenta y los permisos de la sesión del dashboard siguen separados.

La baja comprueba primero la existencia y protección del último administrador,
retira todas las credenciales vinculadas y sus sesiones, y finalmente escribe
el catálogo. Usa una sola admisión retenida hasta completar sus pasos, aunque
el observador cancele o el dashboard selle nuevas operaciones. No presenta las
dos persistencias como una transacción atómica: un fallo posterior al retiro
informa `credentials_retired: true`; un resultado de retiro incierto conserva
`null`. No repite escrituras automáticamente.

La autoridad consulta un snapshot canónico inmutable y sin locks del catálogo.
Las cuentas ausentes, inactivas o no verificables no habilitan acceso personal.
Los retiros durables impiden nuevos bindings y la reaparición de credenciales
antiguas aunque se restaure sólo el JSON. El esquema privado pasa de 1 a 2;
la migración preserva roles y sesiones válidos, y una base 2 sin su tabla de
retiros se rechaza. Una versión anterior del programa rechaza el esquema 2.
Esto no acredita todavía restauración conjunta de backups ni rollback de P6.

La revisión `13d0ec4` pasa 6.285 pruebas con un xfail conocido. El suplemento
`50135f9`, limitado al manejo de errores de la baja y su prueba, pasa otras
60 pruebas de administración y Ruff; las suites se superponen y no se suman.
Se conservaron todos los intentos y hashes de las 770 fuentes verificadas en
`data/autonomous-roadmap-20261102/t047/people-backend-evidence.json`.
Esta unidad prepara el backend de Personas. La vinculación vocal persistente
sigue pendiente; T047 y T013 conservan su estado en curso.

### Personas NEXO con operaciones reales

Personas deja el enlace técnico y abre tarjetas, detalle y formularios NEXO
con el catálogo real. Permite crear una cuenta, cambiar su nivel y confirmar
su baja. Sólo consulta los cuatro escalares administrativos; conserva estados
activos/inactivos y distingue una respuesta vacía de una fuente indisponible.
El rol de la sesión permanece visible y separado del nivel de la persona.

La baja requiere permisos de administración y de credenciales. Crear o cambiar
el nivel no crea accesos ni vincula voces. La interfaz usa texto DOM para nombres
y verifica el catálogo después de las escrituras. No publica resultados
optimistas ni repite una mutación cuando se pierde su respuesta; muestra el
resultado incierto y permite consultar por GET antes de otro envío manual.

El controlador conserva una mutación pendiente al salir y volver a Personas,
descarta GET antiguos y limpia cuentas, selección, formulario y diálogo al
retirar la sesión. Un refresco fallido con un diálogo abierto conserva el
borrador, anuncia el error dentro del diálogo y deshabilita su envío hasta
verificar de nuevo la fuente. Las lecturas del hogar se pausan en esta vista.

Las revisiones `c2cc15a` y `64c01a2` pasan 94 pruebas JavaScript, Ruff y el
recorrido Chromium con tres roles e ingreso HTTP individual. El navegador
comprueba CRUD real, protección del último administrador, retiro de las sesiones
vinculadas, lecturas cruzadas, respuesta perdida sin reenvío, borrado del diálogo
antes del logout y GET antiguo después de otro login. Se verificaron el diseño
de escritorio y el reflow a 320 y 390 píxeles. Las capturas son sintéticas y
están en `data/autonomous-roadmap-20261102/t047/people-browser/`; la evidencia y
los hashes de 771 fuentes están en `t047/people-ui-evidence.json`, bajo el mismo
directorio privado de ejecución. Voces y su contrato de asociación persistente
siguen pendientes. No se acepta todavía ningún criterio de campo ni se cierra
T047 o T013.

### Propiedad de lecturas y recibos de Voces

Cada transporte conserva un namespace opaco del servidor, la autorización
original y su revisión. Dos logins de la misma credencial y del mismo operador
no pueden intercambiar UUID de conexión ni cubrir los recibos del otro.
El namespace se fija antes de aceptar la conexión y encolar datos. REST lo
comprueba antes de la lectura nativa, tras sus awaits, antes del ticket y dentro
del callback que entrega el cuerpo. También retira cursores de páginas vacías
si la conexión o su retención dejó de estar disponible.

Las conexiones, entregas y desafíos de reset guardan ese namespace en SQLite.
La guardia original se comprueba después de adquirir los locks y en el helper
que confirma la transacción, incluidas sus consultas de recuperación. Un ACK
provisional se revierte si la autorización se retira antes del commit.
Desconectar permite liberar sólo la conexión propia, incluso tras revocación;
conserva todos los tickets sin ACK. La migración mantiene pendientes las filas
antiguas sin dueño verificable. El vencimiento tampoco acredita su retirada.

El cierre durante `accept`, la retirada idle sin mensajes y la cancelación del
observador conservan dueños y cleanup. Los fixtures de HTTP, WS y diagnóstico
inyectan una autoridad real; las pruebas de protocolo usan bindings sintéticos
explícitos. Se mantienen separadas las mutaciones administrativas del bearer
anterior y las lecturas privadas que necesitan una autoridad del servidor.

Las revisiones `ba52848` y `8260acb` pasan 6.310 pruebas con un xfail conocido y
Ruff. La primera pasada completa encontró tres fixtures de diagnóstico sin
autoridad; se corrigieron y la segunda pasó sin fallos. Los intentos, revisiones
y hashes de 773 fuentes están en
`data/autonomous-roadmap-20261102/t047/voices-reader-evidence.json`.
Quedan el canal CONTROL_ONLY, su ámbito durable de reset, el controlador NEXO
por socket/login y la asociación manual de perfiles con personas. La lectura
exacta de Evaluación y la interfaz de credenciales continúan en T013. Esta
unidad no acredita restauración conjunta de backups ni aceptación de campo.

### Canal exclusivo de Voces

`/ws/voices` exige lectura y revisión vocal antes de la admisión, y protocolo 3
antes de crear la conexión y el desafío persistidos. El servidor fija el modo
CONTROL_ONLY y sus tres canales: `voices:profiles`, `voices:captures` y
`voices:review`. La suscripción queda fuera del broadcast general. También se
validan cola, ticket, frame y canal REST antes de leer o entregar datos.

El ámbito del reset se añade en el servidor a la clave interna de página,
después de validar el UUID del wire. `technical:UUID` y `voices:UUID` conservan
el namespace de autorización y separan sus obligaciones durables. Ni un reset
vocal cubre deuda técnica ni uno técnico cubre deuda vocal. Las claves antiguas
sin ámbito permanecen pendientes; el cliente no puede reclamar un prefijo.

La revisión `68fbb3d` pasa 877 pruebas de dashboard, herramientas, privacidad,
recibos y revisión vocal, además de Ruff. Comprueba cero proyección, encolado,
bytes y tickets generales para el canal vocal; el suscriptor técnico simultáneo
sigue funcionando. Su UUID no habilita consultas de conversaciones. La retirada
de perfiles emite sólo el control de su canal. Las pruebas de vencimiento
durante accept, idle y envío real cubren ambos modos y páginas vacías.
Los hashes de 774 fuentes y las ejecuciones están en
`data/autonomous-roadmap-20261102/t047/voices-control-evidence.json`.
La suite completa de 6.310 pruebas corresponde a la unidad anterior; sus casos
se superponen con estas 877 verificaciones y no se suman. Continúan el montaje
NEXO de Voces, su retirada real de medios/ACK y la asociación manual con personas.

### Sustitución de sesión del navegador

El login que recibe una cookie vigente sustituye esa sesión en la misma
transacción que crea la nueva. La comparación del predecesor y la guardia de
ambas autoridades llegan hasta el commit; dos reemplazos simultáneos tienen
un solo ganador. Las sesiones de otros navegadores permanecen independientes.
El cupo se comprueba después de retirar el predecesor, permitiendo un reemplazo
uno a uno. Un rechazo anterior al commit revierte ambos cambios; un fallo de
entrega posterior no resucita la sesión retirada.

La revisión `55bee7b` pasa 155 pruebas de sesiones, autoridad, respuestas
privadas retenidas y canales vocales, además de Ruff. Incluye persistencia del
reemplazo, login fallido, carrera simultánea, capacidad y retirada antes de
enviar un cuerpo privado, incluso después de sus headers. La primera ejecución
de esta selección pasó los 155 casos pero falló al cerrar un fixture cuyo cupo
de prueba seguía en cero; se corrigió el alcance del parche y la repetición
terminó con exit 0. La evidencia está en
`data/autonomous-roadmap-20261102/t047/login-replacement-evidence.json`.
Continúan el montaje NEXO de Voces y la asociación manual con personas.

### Voces NEXO con revisión real

La revisión `4d2c0ef` monta Voces en `#voces`, con las columnas, colores y
diálogo del diseño aprobado. Consulta perfiles y capturas reales, reproduce
PCM revisado, confirma perfiles, corrige grupos y permite borrar y recrear un
perfil. Los contadores describen la página actual. El símbolo vocal es
decorativo; los nombres y grupos proceden del ledger. El formulario explica
la conservación del perfil y el vencimiento independiente de audio y texto.

Cada conexión conserva su sesión, socket, stores y controlador originales.
Los GET esperan reset/ACK y `rest_ready`; únicamente se admiten los tres
canales vocales. Salir, vencer o cerrar sesión sella nuevas lecturas y ACK,
limpia DOM, selección, borradores y cursores, y espera los decoders y el
reproductor nativo antes de abrir otra conexión. Un cuerpo tardío no puede
restaurar una vista ni retirar la sesión siguiente. Un 403 funcional provoca
una comprobación de identidad, sin consumir dos veces el cuerpo original.

Un resultado incierto bloquea nuevas mutaciones hasta una actualización real.
La misma confirmación incierta conserva su ID durante la sesión, incluso al
salir y volver a Voces, sin guardar nombres ni muestras para deduplicarla.
El servidor compara el comando completo. Un éxito conocido permite una
intención nueva, incluida la recreación posterior al borrado. La memoria de
IDs inciertos está acotada y se retira al cerrar o cambiar de sesión.

La suite completa pasa 6.346 casos y conserva el xfail conocido de C05/T037,
sin cambios durante su ejecución. El delta posterior de frontend y harness
queda cubierto por 110 pruebas Node, Ruff y Chromium con login individual,
PCM sintético, un `emptied` nativo retenido, pérdida del recibo, recreación,
correcciones, borrado, permisos y reflujo a 320/390 píxeles. La regresión de
Hogar y Personas también pasó. No se suman estas selecciones superpuestas.
Los hashes, logs, XML y capturas están registrados en
`data/autonomous-roadmap-20261102/t047/voices-nexo-evidence.json`.
T047 sigue en progreso: falta la asociación manual persistente voz-persona y
su limpieza al retirar una cuenta. La revisión vocal no concede permisos ni
consentimiento. No se acepta ningún criterio con esta unidad local.


### Asociación manual de voz y persona

La revisión `f25f7c6` conserva el vínculo en una tabla propia, con
revisión independiente del perfil acústico. La API exige revisión vocal y
administración; el vínculo no autentica, concede permisos ni expresa
consentimiento. La lectura enriquecida es explícita y mantiene separados los
campos ordinarios de Voces. Sus recibos verifican la sesión original, la
revisión del ledger y un único snapshot canónico de todas las personas
vinculadas, hasta el envío real del cuerpo.

Retirar una cuenta sella sus credenciales de forma durable, limpia los
vínculos nativos y después permite reemplazar el catálogo JSON. Todos los
helpers comparten una prueba inmutable del mismo ledger; su lectura no toma
locks ni hace SQL bajo el lock de cuentas. Un resultado de commit incierto,
un cierre o una reapertura retiran esa prueba hasta reconciliar SQL. El
borrado del perfil también deja una revisión de desvinculación.

La migración del vínculo y su versión es atómica. Una restauración con tabla
o marcador ausentes, versión desconocida o estructura incompleta se rechaza
antes de reparar esquema. Si ambas tablas faltan se trata como una base
anterior a esta función. Una apertura fallida no publica una conexión parcial.
El pipeline probado con una sugerencia acústica realmente vinculada sigue
usando el contexto compartido anónimo, sin historial, preferencias ni acceso
personal. Se normaliza el ambiente desconocido a `None` al registrar metadata.

La suite completa pasa 6.422 casos y conserva el xfail conocido de C05/T037;
las 784 fuentes permanecieron idénticas. Ruff pasa y la revisión independiente
no conserva bloqueantes. Los resultados enfocados se superponen con la suite.
La evidencia está en
`data/autonomous-roadmap-20261102/t047/person-native-evidence.json`.
T047 sigue en progreso: corresponde aplicar y verificar el selector NEXO con
catálogo auxiliar ligado al dueño de la vista y a su token de entrega. T013
conserva sus pendientes de evaluador y credenciales. No se acepta ningún
criterio ni se modifica producción con esta unidad.


### Selector NEXO de persona vinculada

La revisión `be91b8d` conecta la tarjeta y el diálogo manual de la
referencia con la API real. El selector distingue homónimos por ID canónico
y permite desvincular expresamente. La asociación conserva la revisión
acústica y no cambia permisos ni consentimiento. Un catálogo no disponible
no se presenta como un vínculo vacío; un 409 o 503 exige actualizar sin
reenviar automáticamente ni mostrar un éxito anticipado.

La consulta auxiliar de nombres pertenece al mismo login, socket y vista de
Voces. Sólo admite el GET fijo del catálogo con autoridad administrativa y
retiene su decoder y consumidor hasta su salida real. La aceptación y el
submit comprueban el token de entrega, el perfil y ambas revisiones. Los
objetos anidados se copian y congelan. Al retirar la vista se borran nombres,
opciones y borradores antes de confirmar el DOM vacío. Catálogo, respuesta
de mutación y audio nativo son barreras independientes; su flush conserva el
último token realmente renderizado. Una verificación de identidad diferida
vuelve a comprobar el dueño antes de empezar.

Pasan 120 pruebas Node, Ruff y Chromium con HTTP real, PCM sintético,
homónimos, vínculo/desvínculo, conflicto CAS, indisponibilidad, catálogo
retenido bajo otro token y otro login, borrado de cuenta y conservación del
perfil. También pasa la regresión de Hogar y Personas. Las capturas incluyen
la tarjeta vinculada, el selector y reflujo a 320/390 píxeles. La evidencia y
los hashes están en
`data/autonomous-roadmap-20261102/t047/person-ui-evidence.json`.

La suite de 6.422 casos y un xfail corresponde a `f25f7c6`; los 300 archivos
Python de producto permanecen idénticos. El delta de esta revisión es de
frontend y harness, con la comprobación complementaria indicada. No se
sumaron suites superpuestas ni se atribuyó la suite completa a otro árbol.
T047 conserva pendiente la revisión de la presentación inversa desde
Personas; T013 mantiene evaluador y credenciales. Ningún criterio queda
aceptado ni se cambia el despliegue con esta unidad.


### Consulta de perfiles por persona

La revisión `a813030` añade un filtro canónico opcional a la lectura
enriquecida de perfiles. Exige lectura y revisión vocal más administración.
El SQL selecciona la persona antes de paginar y reunir las fuentes; no lee
vectores, audio ni perfiles ajenos. Cada fila y su fence deben corresponder
al destino. La consulta vacía también conserva la prueba de ese destino.

La revisión se comprueba en el envío nativo y exterior aun sin fuentes, para
que un cursor vacío retenido no sobreviva al retiro de la persona. Una marca
interna, puesta después de validar la proyección vacía, permite comprobar
la autoridad y caducidad de metadata sin relajar la regla raw que exige
fuentes. El plazo de una página vacía de capturas corresponde sólo a esa
consulta: no extiende ninguna captura. Las respuestas vacías sanas mantienen
su conexión y canal.

Pasan 6.443 pruebas y el xfail conocido de C05/T037, con 785 fuentes idénticas
durante la ejecución; Ruff pasa. Las selecciones de 78, 154 y 6 casos se
superponen con esa suite. La primera ejecución encontró una carrera en la
prueba que volvía a buscar una entrada ya retirada; ahora conserva la
referencia previa y comprueba su pérdida de vigencia. La evidencia está en
`data/autonomous-roadmap-20261102/t047/person-filter-evidence.json`.

Corresponde aplicar y probar el widget inverso de Personas, con un único
dueño para ambas vistas, paginación acotada y contadores de página. T047 y
T013 siguen en progreso; ningún criterio ni despliegue cambia de estado.


### Voces vinculadas en el detalle de Personas

La revisión `4c000aa` muestra páginas reales de perfiles confirmados
para la persona seleccionada. El detalle conserva el catálogo y nivel de
cuenta; el widget vocal sólo lee metadata y declara la cantidad de su página.
No obtiene capturas, audio ni nombres auxiliares ni admite mutaciones vocales.
Los nombres se muestran como texto y «Gestionar en Voces» abre la gestión
global de forma explícita, sin prometer un salto a otro registro.

Ambas vistas reutilizan un único dueño de conexión. Cada configuración queda
capturada con persona, host y revisión del detalle. Cambiar selección,
catálogo, host, operación de cuenta o vista sella el anterior antes de limpiar
su DOM. La nueva conexión espera los decoders y consumidores reales. A→B→A
con otro host no adopta datos anteriores; un retry encolado tampoco consume
el timer de otro contexto. Los botones administrativos no reactivan controles
vocales ocupados. El store rechaza filas de otro destino antes de cachearlas.

Pasan 127 pruebas Node, Ruff y Chromium con 51 perfiles reales sintéticos,
páginas 50+1, decoder retenido al cambiar C→B→Voces, vacío frente a 503,
cambio a nivel niño sin alterar vínculo y retiro que limpia los 51 vínculos
conservando los perfiles. La regresión de Hogar y Personas pasa. Las capturas
verifican escritorio y el widget visible a 320/390 píxeles. Los hashes y logs
están en `data/autonomous-roadmap-20261102/t047/person-inverse-evidence.json`.
La suite de 6.443 casos y un xfail corresponde a `a813030`; los 300 Python de
producto permanecen idénticos y el delta de UI/harness tiene la validación
complementaria indicada. No se suman selecciones superpuestas.

La auditoría de la referencia no conserva huecos materiales de T047: queda
`verified_local` para la integración NEXO. T013 sigue en progreso para lector
exacto de evaluación y gestión de credenciales/roles. Ningún criterio C ni P4
queda aceptado y no se modifica el despliegue.

## Lectura nativa de evaluación temporal

El lector nativo admite una muestra y una modalidad expresamente concedidas a
la sesión original de rol evaluador. El permiso operativo de evaluación no
convierte al operador en lector de muestras. Consulta primero la procedencia
acotada y calcula los hashes sobre cada declaración original completa, antes
de convertirla a fuentes tipadas. Rechaza claves JSON duplicadas, fuentes
añadidas o ausentes y cualquier declaración que no coincida con la concesión.

La consulta de audio sólo selecciona WAV y duración del preview; la de texto
sólo selecciona el transcript completo. No leen la otra modalidad, sala, modelo,
embeddings, grupos, perfiles, personas ni historial. Audio conserva el formato
PCM mono de 16 bits a 16 kHz y máximo de seis segundos. Texto limita caracteres
y bytes, incluidos los posteriores a NUL. La validación WAV compartida conserva
el contrato de las capturas operativas existentes.

El vencimiento efectivo es el mínimo exacto de muestra, credencial, sesión,
fingerprint, fuentes y dependencias. No usa el TTL redondeado de la cookie.
Revalida el permiso y la disponibilidad de las fuentes después de terminar
la codificación real. Una revisión de credencial, sustitución de login, cierre
de sesión, borrado pendiente o caducidad impide devolver el contenido.
Los errores esperados se crean fuera de sus manejadores tras limpiar los
locales privados; tampoco conservan el contenido mediante el traceback original.

La prueba inmutable retiene actor, fuentes tipadas reales, hashes completos y
vencimiento originales. Revalidar una proyección mutable no puede cambiar esa
procedencia ni regenerar la prueba. El lector usa el dueño nativo existente de
VoiceRegistry y no construye modelos ni introduce otro registro de tareas.

La suite final pasa 6475 pruebas y una xfail conocida; las 103 pruebas cruzadas
se solapan con ese total e incluyen 32 casos específicos de evaluación. Ruff
no informa errores y los hashes de 789 fuentes permanecieron iguales durante
la ejecución. La revisión independiente está cerrada y la evidencia está en
`data/autonomous-roadmap-20261102/t013/evaluation-native-evidence.json`.
T013 sigue en curso: falta conectar la respuesta HTTP singular, su canal de
control exclusivo, recibos y guardas del envío exterior antes de distribuir
credenciales de evaluación. T047 conserva su verificación local y no se acepta
ningún criterio doméstico o de campo por esta unidad.

## Entrega HTTP y control exclusivo de evaluación

La lectura singular de `/api/evaluation/samples/{fingerprint_id}` exige una
modalidad explícita y una sola cabecera de conexión. No ofrece catálogo ni
cursor y rechaza queries de personas, fuentes o vencimientos. Construye el
actor desde el lease original admitido, sin adoptar otro login. El operador
no se convierte en evaluador por disponer del permiso de evaluación.

`/ws/evaluation` exige rol evaluador antes de subscribe/accept, protocolo 3
y únicamente el canal `evaluation:sample`. Sus conexiones no reciben ni
proyectan publicaciones generales. Los canales y formatos de Voces y del
modo técnico se conservan. Los prefijos de modo y namespaces de login
mantienen separadas las obligaciones duraderas de limpieza de cada página.

La respuesta valida el actor, la fila cerrada y la prueba inmutable original
antes de transferir ownership. Compara fuentes, hashes y vencimiento y llama
al validador desde la clase. El ticket se persiste antes de headers. La guarda
conserva sus valores originales aunque termine la respuesta interior; el dueño
exterior la vuelve a comprobar junto con fuentes y conexión antes de enviar.
El scope cubre middleware y buffers y aplica `no-store` también a errores.

Las pruebas con sesiones, SQLite y sockets ASGI reales cubren otra cookie
del mismo credential, otros modos, formato de DTO alterado, caducidad de la
muestra con sesión vigente y borrado de la fuente dentro de un buffer. Un ACK
temprano no oculta un callback de body aún activo; cancelar al observador
tampoco retira una lectura nativa antes de su salida real. El socket con cookie
y el GET sólo comparten conexión cuando pertenecen al mismo login original.

La suite final pasa 6527 pruebas y conserva una xfail conocida. Las 501 pruebas
cruzadas se solapan con ese total e incluyen 37 casos nuevos de HTTP/control.
Ruff no informa errores y los hashes de 793 fuentes permanecieron iguales
durante la ejecución. Las revisiones independientes están cerradas; evidencia:
`data/autonomous-roadmap-20261102/t013/evaluation-http-evidence.json`.
T013 sigue en curso: falta la emisión, listado, revisión y revocación segura
de credenciales y la navegación por rol. La siguiente unidad preparará proofs
de fuentes reales y una identidad duradera del intento antes de exponer tokens
de un solo envío. T047 conserva su estado y no se acepta ningún criterio por
esta verificación local.

## Preparación nativa de concesiones de evaluación

VoiceEvaluationReader prepara concesiones desde metadatos reales y la
existencia de la fila de review. No consulta audio, transcript, embeddings,
modelos, personas, grupos ni perfiles. Conserva los hashes de las declaraciones
originales completas y limita cada muestra por los vencimientos solicitados,
de credencial, fingerprint, fuentes y dependencias. La duración de la sesión
emisora no limita automáticamente una concesión emitida mientras tenía autoridad.

PreparedEvaluationScope es inmutable y conserva el emisor original, las muestras
y sus fuentes tipadas. Usa una referencia débil al ledger y una generación de
apertura monotónica. Cerrar y reabrir el mismo objeto no revive la preparación,
incluso si un ledger de memoria vacío vuelve a considerar vigentes las fuentes.
Su predicado puro comprueba fuentes, generación y ambos relojes antes y después
de la consulta, sin volver a exigir `issuer.current()`.

Authority acepta opcionalmente ese proof tipado, exige actor, rol, muestras y
vencimiento coincidentes antes de generar el token o sellar una revisión y
lo revalida en la guardia nativa final. Una revisión propia conserva la guardia
especial del emisor original. Un binding nativo inválido se rechaza antes del
sello. La compatibilidad nativa con `prepared_scope=None` se conserva; la futura
emisión HTTP no debe usarla como fallback para concesiones de evaluación.

Las pruebas usan authorizer SQLite para impedir toda lectura de payload y
cubren plazos exactos, cambios de fuentes en el commit, revisión propia,
mismatches sin retirar leases previos, reloj atrasado, cierre/reapertura y
cancelación del observador esperando la salida real del trabajo retenido.
La suite final pasa 6558 pruebas y conserva una xfail conocida. Las 216 pruebas
cruzadas se solapan con ese total e incluyen 31 casos nuevos. Ruff no informa
errores y los hashes de 795 fuentes permanecieron iguales durante la ejecución.
Las revisiones independientes están cerradas; evidencia:
`data/autonomous-roadmap-20261102/t013/credential-proof-evidence.json`.

La siguiente unidad de T013 añadirá el journal duradero específico de comandos
de credenciales, CAS y catálogo seguro. El marcador debe sobrevivir a revocación
y purga; replay devuelve metadata y no genera ni recupera otro secreto. Antes
de exponer emisión HTTP faltan esa reconciliación y ownership compartido con
los writers nativos existentes. No se emitieron credenciales reales ni se
aceptan criterios domésticos por esta unidad; T047 conserva su verificación local.

## Comandos nativos de credenciales y reconciliación

NativeCredentialCommands reúne emisión, revisión y revocación bajo una intención
UUID canónica, el namespace del emisor original y el hash del comando solicitado.
El journal de schema3 registra en la misma transacción el efecto y su recibo
terminal, committed o no_effect. Sus 4096 marcadores no se purgan al vencer o
revocar una credencial; al llenarse rechaza nuevas intenciones sin desalojarlos.
Un replay exacto devuelve metadata antes de preparar fuentes o generar secretos.
Cambiar el comando o la sesión emisora produce conflicto.

El secreto sólo lo recibe el primer observador de una emisión aplicada. No queda
en el journal, catálogo, recibo, repr ni resultado del task propietario. Cancelar
ese observador cierra su slot y conserva el trabajo real; un replay no recupera
el secreto perdido. Catalog y attempt requieren auth:manage vigente y devuelven
metadata separada. Otro login administrativo puede reconciliar un recibo seguro;
repetir la mutación sigue exigiendo el namespace original. HOME y EVALUATOR no
obtienen esta administración.

CAS exige la revisión esperada y comparte reservas por target con los writers
nativos existentes. Cada propietario libera sólo su reserva al completar la
publicación o probar un rechazo previo al commit. Las marcas ambientales de
retiro de cuenta y revocación permanecen independientes. Las concesiones de
evaluación de esta fachada requieren PreparedEvaluationScope real, obtenido
desde metadatos, y revalidan selección, modalidad, fuentes y plazos en la guardia
nativa final. Una guardia perdida revierte el savepoint y persiste no_effect con
un snapshot releído. Una revisión propia conserva al emisor original.

Entrar en commit vuelve incierta cualquier excepción posterior, incluida una
excepción dentro de commit después de confirmar o durante la publicación.
Authority cierra su disponibilidad y conserva la reserva; reabrir permite leer
el recibo duradero sin regenerar el secreto. Cancelar el task propietario, aun
antes del despacho, también cierra la autoridad. El observador público permanece
protegido por shield y no cancela ese propietario.

La migración de los esquemas 1 y 2 al 3 es atómica. Una base del esquema 3 debe conservar sus tablas,
columnas, índice único de digest, claves temporales y recibos semánticamente
válidos. El reloj crudo se valida antes de combinarlo con el actual, y ninguna
emisión, sesión, retiro o recibo puede superar su reloj persistido. El cruce
del journal impide que una restauración parcial reactive un target revocado o
una revisión inferior ya retirada. No se afirma detectar el rollback completo
de una base sin un ancla externa monotónica.

La suite final pasa 6620 pruebas y conserva una xfail conocida. Las 153 pruebas
cruzadas se solapan con ese total e incluyen 62 casos nuevos. Ruff no informa
errores y los hashes de 800 fuentes permanecieron iguales durante la ejecución.
Las revisiones independientes están cerradas. La primera corrida completa falló
en tres fixtures antiguos y cuatro pruebas de temporización; los fixtures se
corrigieron sin relajar los controles y los cuatro casos pasaron en repetición.
Una segunda corrida quedó incompleta; la corrida final completa terminó con
exit0 mediante un ejecutor separado y acotado. La evidencia enlaza todos los logs:
`data/autonomous-roadmap-20261102/t013/credential-commands-evidence.json`.

La siguiente unidad de T013 conectará esta fachada con HTTP y la entrega
revocable del secreto inicial, y después con Accesos y las vistas de rol NEXO.
Esta unidad no expone emisión HTTP ni acepta criterios domésticos. T047 conserva
su verificación local; no hubo cambios de producción.

## Administración HTTP de credenciales

Cinco rutas bajo `/api/admin/auth` exponen el catálogo de credenciales, la consulta
de `credential-attempts` y los comandos POST, PUT y DELETE. Todas requieren
auth:manage, Authority real y los propietarios de trabajo y entrega. La ausencia
o invalidez de esas dependencias impide operar credenciales. El autenticador
global conserva su contrato; la exigencia nativa se comprueba antes del cuerpo
y de los comandos, sin habilitar operaciones mediante el modo legacy standalone.
Personas, cuentas canónicas, perfiles vocales y DashboardRole siguen
siendo autoridades separadas.

El cuerpo JSON tiene límite de 64 KiB y campos cerrados. Rechaza claves
duplicadas, números no finitos, revisiones fuera del rango nativo, selecciones
duplicadas y texto que no pueda codificarse en UTF-8. Los errores 422 son fijos
y no reflejan valores recibidos. No se exige una fecha futura al parsear: una
intención ya terminada debe poder reconciliarse después de su vencimiento.
Las primeras ejecuciones siguen sujetas a las guardias temporales nativas.

La emisión de evaluación usa un proveedor diferido con VoiceRegistry y
PreparedEvaluationScope real. Comprueba que el ledger corresponda al propietario
de la entrega. El journal se consulta antes de preparar fuentes; el cliente no
puede aportar hashes de fuentes ni usar la compatibilidad nativa sin proof.
GET devuelve metadatos; sólo la primera emisión aplicada incluye el secreto.
Un login administrativo nuevo puede consultar el recibo, pero no repetir la
mutación bajo el namespace de otra sesión ni recuperar el secreto.

CredentialResponse conserva el emisor original y vincula una sola vez el lease
del secreto emitido, con la revisión exacta del recibo. Ese guardia permanece en
el scope exterior tras finalizar la respuesta interna. Verifica sesión, target
y cierre del propietario antes de headers y de la entrega real del cuerpo,
incluso detrás del buffering del framework. Un target revocado o revisado entre
commit y lookup no puede convertirse en un lease posterior.

La secuencia preparar, ejecutar, serializar y enviar queda retenida por el
propietario administrativo y RestDeliveryOwner; el task devuelve únicamente
RestResult. No usa tickets, snapshots ni ACK de Voces para completar operaciones
administrativas. Los mensajes y secretos se limpian en las salidas de error,
incluida la capacidad del transporte. Cancelar el observador no cancela el
trabajo nativo ni convierte un replay en una nueva emisión.

La suite completa pasa 6705 pruebas y conserva una xfail conocida. Los 65 casos
específicos de HTTP pasan; la matriz general incluye además las cinco rutas
nuevas. Ruff no informa errores y los hashes de 803 fuentes permanecieron
iguales durante la corrida final. Ambas revisiones independientes están
cerradas. La primera corrida detectó 15 diferencias de precedencia de
validación; se corrigió su ubicación preservando el contrato global y se añadió
la matriz con Authority real y el rechazo de bearer legacy sin esa autoridad.
Los resultados y logs se conservan en
`data/autonomous-roadmap-20261102/t013/credential-http-evidence.json`.

La siguiente unidad añadirá el catálogo acotado de muestras del propio
evaluador y conectará Accesos y la vista de evaluación NEXO con estas rutas.
El refactor visual T047 conserva su verificación local. No se despliega
producción ni se aceptan criterios domésticos mediante esta unidad.

## Catálogo de concesiones del evaluador

DashboardAuthority.evaluation_grants proyecta únicamente las concesiones del
lease original de EVALUATOR, con evaluation:read vigente y de la misma
Authority. Operator no accede a este catálogo aunque posea evaluation:read.
El getter usa el snapshot de autorización y no consulta SQLite, VoiceRegistry,
UserManager ni contenido de muestras.

EvaluationGrantMetadata y EvaluationGrantCatalog son DTOs congelados. La
proyección pública contiene observed_at, expires_at y hasta 64 filas con
fingerprint_id, modalidades y vencimiento efectivo. No publica hashes, fuentes,
nombres, Personas, perfiles, audio ni transcripciones. Cada fila se limita por
su muestra y el lease; el catálogo vence con su primera fila. Un catálogo vacío
vigente mantiene el límite finito de la sesión. La proyección nunca sustituye
una sesión retirada por una lista vacía.

GET /api/evaluation/grants rechaza parámetros de consulta y requiere Authority
y propietario de entrega reales. No necesita un canal WebSocket ni tickets.
Su scope conserva sources=(), private_envelope=False y el guardia original en
la frontera exterior de HTTP. Las muestras originales, la sesión y el deadline
se revalidan durante headers y buffering; el guardia sobrevive a la limpieza
de la respuesta interna. Un catálogo retenido que vence se retira completo.
Una petición nueva podrá devolver las concesiones restantes.

El catálogo declara autorización, no disponibilidad del contenido. Una fuente
retirada puede seguir figurando dentro del plazo concedido; el lector singular
continúa comprobando existencia, modalidad y fuentes antes de entregar datos.
El getter conserva el floor temporal en RAM y no añade persistencia: la frontera
HTTP mantiene synchronize y las reglas de cierre de Authority existentes.

La suite completa pasa 6750 pruebas y conserva una xfail conocida. Pasan 472
casos cruzados, incluidos 41 nuevos; la matriz general también cubre la ruta.
Ruff no informa errores y los hashes de 805 fuentes permanecieron iguales
durante la corrida completa. Las revisiones independientes están cerradas.
La prueba de contenido retirado usa el contrato LookupError del lector nativo;
la expectativa inicial se corrigió sin modificar ese comportamiento. Evidencia:
`data/autonomous-roadmap-20261102/t013/evaluator-catalog-evidence.json`.

La siguiente unidad integrará el selector automático y las vistas de Accesos y
Evaluación en NEXO. Los borradores de frontend se preparan fuera de src mientras
se verifica este backend. No se modifican producción ni criterios de aceptación
doméstica con esta unidad.

## Accesos y evaluación en NEXO

La interfaz incorpora Accesos para operator con auth:manage y Evaluación para
evaluator con evaluation:read. El evaluador carga automáticamente su catálogo
de concesiones y consulta una muestra y modalidad sólo por acción explícita.
No crea la casa 3D ni consulta Hogar, Personas o Voces. La estética, navegación
y presentación adaptable proceden de la referencia NEXO aprobada en T047.

Accesos usa el catálogo, los comandos CAS y los recibos del backend real.
Cada gesto fija un request_id y un cuerpo antes del envío; un resultado incierto
se consulta por recibo y no repite la emisión. La credencial se entrega una vez
al panel visible, enmascarada, con copia explícita; nunca entra en snapshots,
almacenamiento del navegador, URLs ni recibos. Cerrar, ocultar o sustituir la
sesión elimina el panel. El catálogo admite paginación y distingue fallo de una
lista vacía. El selector de Personas conserva IDs canónicos y desambigua homónimos.

Voces permite preparar una concesión desde muestras vigentes y modalidades
elegidas explícitamente. Transfiere sólo IDs y límites temporales a Accesos;
primero retira la revisión de audio. La importación abre un borrador: emitirlo
requiere otra confirmación. La cuenta, su nivel, el vínculo manual perfil-persona,
el rol del dashboard y el alcance de evaluación siguen siendo independientes.

El propietario de Evaluación usa el canal exclusivo, recibos y lector singular
ya verificados. Sólo una modalidad ocupa la vista. Cambiarla, ocultar la pestaña,
revocar o cerrar la sesión retira DOM, Blob y audio; el siguiente contexto espera
el emptied y estado nativo del decodificador. No hay ACK por timeout. Una limpieza
visual fallida todavía retira los recursos y bloquea la reapertura. La navegación
comprueba generación, rol y visibilidad después de cada barrera; una respuesta
anterior no recupera la vista. Un 403 funcional verifica la sesión original.
El logout con 401 reconoce que la sesión ya no es válida y permite otro login;
un error de transporte conserva el cierre pendiente.

Pasan 160 pruebas de frontend y Ruff. Chromium real contra servicios locales
con datos sintéticos cubre HOME/OPERATOR, modificación y revocación CAS, emisión
única, pérdida de recibo sin POST duplicado, selección de dos muestras con
modalidades distintas, consulta aislada del evaluador, audio/transcripción,
retirada con emptied retenido, revocación durante reproducción y nuevo login.
También pasan las regresiones de Hogar/Personas y Voces, incluida la paginación
de 51 perfiles. Hay capturas de escritorio y anchuras 390/320 sin desborde de página.

La ejecución en navegador detectó y corrigió los campos required ocultos para
HOME/OPERATOR, el estado loading que bloqueaba el selector y la salida de una
sesión ya revocada. Las revisiones añadieron cobertura de Unicode por puntos de
código, revisiones exactas y fallo de limpieza. Se conserva la evidencia de los
intentos anteriores; no se presentan como validaciones finales.

Evidencia: `data/autonomous-roadmap-20261102/t013/nexo-ui-evidence.json`.
Los 309 archivos Python de producto coinciden con el snapshot del backend
verificado (6750 pruebas y una xfail conocida); esta unidad cambia frontend y
pruebas de navegador. No se repite esa suite como si validara el frontend.
T013 sigue en curso, T047 permanece verified_local, no se acepta ningún criterio
doméstico ni se despliega en producción.

## Cierre local de sesiones y autoridades

La revisión de T013 contrasta su alcance con ACCESS-01, AUD-01 y los contratos
de H1/H3. La matriz actual verifica rechazo HTTP/WS antes de consumir servicios,
roles del servidor, identidad del login original, límites de tiempo, revocación,
operaciones nativas retenidas, entrega privada y evaluación por muestra y modalidad.
La voz anónima conserva sólo el ámbito doméstico compartido; una asociación
vocal no habilita historial, preferencias ni autoridad personal.

La auditoría encontró una obligación pendiente de ACCESS-01: explicar cómo
obtener el acceso. La entrada NEXO ahora indica pedir una credencial de KZA al
administrador del hogar, solicitar evaluación temporal para revisar muestras y
no usar una credencial de Home Assistant. La explicación está vinculada al campo
mediante aria-describedby; Chromium verifica el contenido y el recorrido de login.

Pasan 956 casos actuales de autoridad, sesiones, adaptadores nativos, comandos
de credenciales y evaluación. No se suman a las 6750 pruebas anteriores como si
fueran casos distintos. Los 309 archivos Python de producto siguen iguales al
backend previamente verificado. También pasan el recorrido NEXO de Hogar/Personas
y Ruff. Los avisos de deprecación de datetime y el overflow provocado por una
entrada negativa se conservan en el log; no son fallos de esta unidad.

T013 queda verified_local. Esto no demuestra acceso a grabaciones reales actuales
ni acepta C02, C04 o C07: siguen pendientes las matrices de aceptación, el inventario
y retiro global, la auditoría completa de accesibilidad y la evidencia de campo.
T014 conserva la procedencia técnica y los intervalos; T033/T035 conservan sus
recorridos completos. Evidencia: `data/autonomous-roadmap-20261102/t013/closure-evidence.json`.

## Procedencia y comparación exacta del corpus

La primera unidad de T014 añade CaptureProvenance, DSPObservation y FrameInterval
al modelo privado de evaluación. Distingue origen, entrada opaca, rama, canal,
tasa, generaciones de apertura/segmento y cantidad de frames de la PCM poseída.
Los estados DSP desconocido o declarado no acreditan una revisión observada.
Las etiquetas históricas channel/dsp_revision no completan esos datos.

El esquema 1 conserva el inventario previo y rechaza campos de comparación nuevos.
El esquema 2 admite CorpusComparison sólo con un audio, una referencia humana
revisada y una hipótesis STT ligados explícitamente al mismo artefacto, captura
e intervalo de frames. No calcula una intersección ni recorta texto para hacerlos
coincidir. Los milisegundos son una envolvente de compatibilidad; el final real
de la sesión se contrasta con la cantidad de frames y la tasa mediante aritmética
entera, incluyendo extremos fraccionarios respecto al reloj de microsegundos.

Un hijo con frames conocidos requiere padres con frames conocidos y contención
exacta. La admisión revalida los tipos de registros y las condiciones de la sesión,
sin confiar en una comprobación anterior. La comparación hereda el vencimiento
original, cierre y revocación; no incorpora un plazo propio. Un registro anterior
sin procedencia puede seguir inventariado, pero no se convierte en comparable.

Los 120 casos enfocados cubren datos anteriores, tipos inválidos, versiones,
DSP declarado/observado, límites exactos, linaje, consentimiento y retirada.
Las revisiones independientes detectaron el padre sin frames, sesiones sin
revalidar y un rechazo por redondear el final de grabación. Los casos que
reproducen esos fallos se conservan junto con sus correcciones.
La revisión temporal también detectó consentimiento posterior a la captura
durante una repetición de hora local. El orden y los límites se comparan ahora
como instantes UTC. La corrida completa anterior se interrumpió antes de editar
fuentes; sus 1750 casos completados no se presentan como una suite completa.

La repetición completa pasa 6838 pruebas y conserva una xfail conocida en
497,13 segundos; los hashes de 837 archivos de fuente y pruebas permanecen
iguales. Ruff no informa errores. Evidencia:
`data/autonomous-roadmap-20261102/t014/corpus-contract-evidence.json`.

Este modelo comprueba declaraciones del manifiesto; no observa el hardware,
lee archivos, calcula WER ni otorga autorización runtime. La siguiente unidad
transportará las dos ramas ambientales y la rama de comandos con la PCM real,
conservando first-writer, vencimiento, borrado y source_digest. T014 permanece
en curso y no habilita grabaciones ni acepta C04/C12 por sí solo.

## Metadatos nativos de las muestras de voz

La siguiente unidad de T014 prepara el registro y la persistencia de procedencia.
VoiceCaptureReview conserva dos descriptores y sus intervalos: el audio usado
para embedding/preview y la entrada que produjo el texto STT. Pueden declarar
canales distintos. El truncamiento de texto es un dato separado, verdadero,
falso o desconocido; no se deduce de la longitud del texto retenido.

Los codecs usan documentos JSON v1 cerrados de hasta 4096 bytes UTF-8, separados
por modalidad. Rechazan duplicados, tipos/versiones inválidos, pares incompletos
y contenido no finito. Los errores conservan un código fijo y eliminan el
contenido del parser de sus tracebacks. SQL NULL conserva el estado desconocido;
metadata no nula inválida deniega la revisión, sin convertirse en legacy.

El constructor fija el preview desde la PCM realmente poseída y valida su WAV
canónico; no reconstruye frames de una duración redondeada. La entrada STT
declarada cubre su segmento completo y no adopta el intervalo del preview.
VoiceRegistry valida y copia los descriptores antes de admitir el trabajo,
conserva el propietario nativo existente y los retira al terminar. El cierre
espera la salida real del modelo; no deja un commit tardío autorizado.

La migración de voice_capture_reviews añade audio_metadata_json y
transcript_metadata_json con un marcador propio de versión 1. La forma antigua
de seis columnas se migra atómicamente sin actualizar datos anteriores. Una
forma parcial, un marcador perdido con columnas nuevas, una tabla perdida con
marcador o una versión desconocida fallan antes de la reparación genérica.
Un fallo de DDL revierte su migración y el inicializador no publica conexión.

La escritura revalida el DTO y usa columnas explícitas en la misma transacción
que el preview y texto. La primera escritura conserva su contenido y plazo;
un duplicado no rellena procedencia desconocida. La duración de una descripción
tipada debe coincidir frame por frame con la captura; la tolerancia histórica
sólo permanece para registros sin esa descripción. La cascada y el vencimiento
existentes retiran la metadata con la muestra.

La revisión limita ambas columnas antes de devolverlas a Python, decodifica
fuera de SQLite y vuelve a comprobar actor y fuentes. Conserva las claves de
la respuesta pública actual. Pasan 318 casos cruzados, incluidos migración,
codecs, propiedad del trabajo, cierre, revisión anterior, vínculos y evaluación.
La suite completa pasa 6977 pruebas y conserva una xfail conocida en 508,94
segundos. Los hashes de 841 archivos permanecieron iguales y Ruff pasa.
Evidencia: `data/autonomous-roadmap-20261102/t014/review-native-evidence.json`.

La ausencia simultánea del marcador y toda la tabla no distingue una base
previa a esta función de un rollback completo sin ancla externa. Esta unidad no añade
esa garantía ni modifica fuentes, hashes de concesiones o retención. Sigue
pendiente conectar los productores reales y exponer metadata por modalidad
con validadores frontend coordinados. T014 continúa en curso.
La ampliación de columnas requiere un runtime compatible; esta verificación
local no acredita un rollback de la instalación con escritores antiguos.
La prueba de migración en copia y el rollback del candidato siguen en P6.
