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
