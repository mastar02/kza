# H1: control autorizado y semántica segura

H1 implementa R01, R02, R03 y consentimiento de R08 del
[roadmap](2026-10-25-stabilization-roadmap.md). Parte del
[baseline y contratos de H0](2026-10-25-h0-baseline-contracts.md).
H0 todavía necesita evidencia viva y resolver el tratamiento y la migración en
copia de memoria legacy; su vista previa privada ya existe. Se avanza con
contratos y pruebas sintéticas sin habilitar dispositivos ni modificar producción.

## Secuencia y criterios

| Frente | Trabajo y evidencia de cierre | Estado |
|---|---|---|
| Semántica | Corpus de pasado, estado, preguntas informativas, negación, condición, reporte y diferidos. Gate, gramática, evidencia del LLM y routers conectados deben rechazar; los pedidos inmediatos conservan dirección e infinitivos/cortesía válidos. | Primera implementación y regresiones sobre texto completo. |
| Captura parcial | Early-dispatch es preparseo especulativo: espera el endpoint, mantiene el texto ligado al audio y nunca ejecuta un prefijo. Una captura cortada por timeout no es una orden completa. | Implementado y probado con ambos routers y modo ambiental; validación acústica real en H6. |
| Sesión | HTTP y WebSocket privados comparten autenticación cerrada sin token, incluyendo listas, recordatorios, presencia, observabilidad y `/ws/live`. Orígenes inválidos se rechazan con sesión válida. Login y recursos estáticos tienen excepciones explícitas. Bind loopback detrás del proxy y token vacío en `.env.example`. | Implementado y probado; proxy y unit reales se validan en H6. |
| Política | Un servicio de acciones y un actor de procedencia por productor. Política obligatoria después de rewrites y antes de todo envío; permisos revisados también al disparar acciones diferidas. Tests sin hooks, con rewrite y revocación. | Frontera, NLU y expansión conectados con pruebas sintéticas; falta evidencia del hogar real. |
| Operaciones | Extender DTOs, validación y confirmación a clima y cover. Agregar temperatura de color para reproducir las escenas seleccionadas sin perder parámetros. Usar rangos, modos y entidades del inventario privado aprobado. | Contratos, servicio, persistencia, formularios e interpretación determinística de voz implementados. Expansión integrada; falta validar el hogar real. |
| Grupos y escenas | Expandir definiciones verificadas a hojas autorizadas y operaciones tipadas. Rechazar macros opacas, templates sin contexto y efectos indirectos no verificados. Pruebas de escena que oculta lock/alarm, grupos no autorizados y parámetros desconocidos. | Implementado con grupos/escenas sintéticos y todos los productores; revisión real pendiente. |
| Consentimiento | Destino cloud explícito, independiente de host/transporte. Denegación probada en reasoner, compactor y cada consumidor; loopback no elude consentimiento. | Implementado; validación offline conectada y límites descritos abajo. |

El actor autenticado se obtiene de la sesión del servidor. Voz, SpeakerID y
agrupación acústica no aportan identidad autenticada. La selección del usuario
está registrada fuera de Git para la futura release; no se activan ahora sus
entradas ni se toma el snapshot persistido como prueba de disponibilidad actual.
Los tipos cover permanecen dentro del alcance aunque todavía no haya una entidad
cover identificada para seleccionar.

## Primera implementación semántica

`tests/command_corpus.py` contiene frases sintéticas etiquetadas sin audio,
personas ni IDs del hogar. Se comparte entre las pruebas de contrato de NLU y
las de `RequestRouter`, con wake acústico confirmado y sin exigir wake textual.
Los casos rechazados se ejercitan en ambos modos del router y verifican que no
se llega al orquestador, búsqueda vectorial ni envíos HA. Los pedidos inmediatos
se comprueban además atravesando el router orquestado.

`non_immediate_action_reason` conserva el bloqueo común de negación, condición
y reporte. Agrega tiempos pasados/perfectos/futuros, marcadores de pasado,
consultas informativas y descripciones de estado. Distingue «apagué» de
«apague», y mantiene «subí»/«abrí» como imperativos salvo contexto de pasado.
La misma regla protege la evidencia atribuida al LLM: su confianza no convierte
una pregunta rechazada en una acción ni habilita otro intent.

El corpus expuso una inversión adicional: «apagá la luz que está prendida»
resolvía `turn_on` porque el stem `prend*` tenía prioridad. Los participios de
estado dejaron de servir como verbos de acción, tanto para la gramática como
para justificar la salida de un LLM. Se prueba la dirección y el rechazo de la
acción opuesta. Los ecos de estado sin orden siguen rechazados; una orden
binaria explícita conserva su validez aunque mencione «luz encendida».

Validación: 40 frases rechazadas y 21 pedidos inmediatos, con 164 regresiones
nuevas entre contrato de NLU, dirección del LLM y routers. La suite completa
produce **3287 passed, 10 failed, 1 xfailed**, en 80,37 s: conserva exactamente
los diez IDs de fallo clasificados en H0. Ruff pasa. El
[registro](../research/2026-10-25-h1-semantic-validation.json) conserva alcance,
comandos y limitaciones; no es una validación completa de H1.

Ese primer registro cubría texto completo entregado al router. La corrección
posterior de captura se describe a continuación. Las pruebas no sustituyen STT
acústico, FAR/FRR ni evaluación de las pausas reales del habla.

## Finalidad de captura y procedencia del texto

El worker de preparseo continúa observando el audio hasta el endpoint; la
presencia de intent y entidad no dispara una acción. Su resultado se asocia a
una generación de captura para evitar que una inferencia pendiente se publique
sobre la siguiente captura. El loop cancela y recoge los workers al terminar.

`CapturedTranscript` liga texto y productor al contenido exacto de un snapshot
mono float32. El router sólo lo reutiliza si coincide con el audio final; el
digest es efímero y no se persiste. `wake_text` y el parser parcial sin cobertura
son diagnósticos, no reemplazan STT. Una continuación, aun con el mismo número
de samples, obliga a evaluar la transcripción del audio final.

Los loops distinguen silencio de timeout. Conservan frames silenciosos para
cerrar VAD, requieren la cantidad de silencio configurada y marcan incompleta
una captura forzada por duración. Los eventos early antiguos sin finalización
explícita también se rechazan antes de STT. Ambient conserva los segmentos
cortados para almacenamiento, pero no los despacha como comandos; el wake textual
reutiliza sólo texto ligado al audio de su segmento completo.

Las pruebas conectadas ejercitan prefijo válido → continuación («mañana»,
negación o condición) → endpoint → gate → ambos routers, sin llamadas HA. También
prueban la generación durante una inferencia pendiente, cambios de audio con
igual longitud, el loop single-room y la conservación de segmentos ambientales
incompletos sin invocar wake. La pausa de VAD sigue siendo una decisión acústica:
sus thresholds y la respuesta no cacheada se evalúan con equipo real en H6.

## Acceso privado uniforme

`DashboardAPI` aplica una dependencia de sesión por defecto, también a rutas
nuevas. Observabilidad añade la misma frontera al registrarse por separado.
Las excepciones de aplicación son GET `/` (shell sin datos) y POST
`/api/admin/auth/login`; los mounts públicos sirven únicamente recursos estáticos.
Los endpoints privados, incluido health, requieren sesión. No se publican las
rutas automáticas de documentación/OpenAPI. Logout requiere sesión.

Sin token configurado, HTTP responde 503 y los sockets rechazan el handshake;
sin credencial válida, HTTP responde 401. Un Origin presente debe coincidir con
esquema, hostname y puerto efectivo, tanto en lecturas como mutaciones y sockets;
orígenes `null`, malformados o con userinfo/path se rechazan. Los clientes nativos
pueden omitir Origin. Las cookies son HttpOnly y SameSite=Strict; usan Secure
cuando la solicitud llega como HTTPS desde el proxy de confianza.

El YAML candidato liga el dashboard a `127.0.0.1:9500` y `.env.example` declara
`KZA_DASHBOARD_TOKEN` vacío. El proxy de la única release debe preservar Host y
establecer el esquema externo mediante headers de forwarding confiables; no se
acepta Origin como sustituto de la credencial. No se modificó el proxy instalado
ni se reinició producción.

La UI de observabilidad comprueba la sesión antes de cargar datos o abrir WS.
Al salir o perder autenticación, cancela cargas, cierra WS y limpia datos
privados; una respuesta tardía no reactiva la sesión. Los rechazos de sesión no
se convierten en datos simulados. Las seis pruebas Node de ese ciclo se ejecutan
también en un job de CI. El JSX modificado se validó con el Babel ya fijado por
la página, verificando antes su SRI.

La matriz de pruebas enumera todas las rutas HTTP privadas registradas y prueba
ausencia de credencial, credencial inválida, configuración ausente y origen ajeno,
sin usar los servicios inyectados. Los dos WS se prueban antes de suscribir o
aceptar clientes. Los tests funcionales existentes usan sesiones reales de prueba,
sin desactivar la dependencia. El test obsoleto que esperaba crear usuarios sin
token se reemplazó por 503 y ausencia de escritura: queda cerrado ese fallo del
rescate; los otros nueve siguen explícitos.

Verificación del commit `6b8d143`: **3549 passed, 9 failed, 1 xfailed** en
103,38 s. No aparecen IDs de fallo nuevos. Ruff 0.15.1 pasa y Bandit 1.9.4 no
encuentra hallazgos de severidad media o alta. Las seis pruebas Node, el smoke
autenticado del dashboard y el smoke compilado (incluyendo la nueva sesión JS y
los rechazos de health sin credencial/configuración) pasan. El
[registro de captura y acceso](../research/2026-10-25-h1-capture-auth-validation.json)
distingue estas pruebas de las comprobaciones acústicas, del proxy instalado y
de los demás frentes de H1 que permanecen pendientes.

## Destino de procesamiento y consentimiento

`EgressPolicy` liga un destino declarado y un consentimiento booleano estricto
a un endpoint. Sin declaración, el destino es cloud; una URL loopback no prueba
dónde procesa el backend. `local` es una declaración del operador sobre ese
backend, que debe contrastarse en la validación del entorno de release.

Los consumidores comprueban esa política antes de enviar, también con un SDK
ya creado: HTTP reasoner, FastRouter, CLI Hermes, cards del índice, destilación
ambiental, generador de frases del inventario y providers remotos. Los clientes
HTTP no siguen redirects; un cambio de endpoint requiere su propia política.
Una URL ausente se rechaza antes de que el SDK elija su proveedor por defecto.
No se eligió otro proveedor ni se cambió el consentimiento configurado.

La compactación hereda el destino del reasoner; un override necesita su propia
declaración. Si falta consentimiento cloud, puede usar el 7B local ya existente,
sin heredar credenciales cloud. El fallback del gateway sigue siendo cloud,
aunque se resuelva a loopback. Ambient exige procesamiento local y los providers
STT/TTS no heredan la autorización cloud del reasoner para enviar audio.

Las pruebas ejercitan consumidores reales con transportes falsos: denegación
antes del SDK/subproceso/sesión, clientes en caché, cambio de política, resolución
de placeholders, cadena de failover y factory del índice. El SDK OpenAI real,
con transporte HTTPX en memoria, recibe un redirect y no lo sigue. No se envían
conversaciones, código ni audio a servicios reales para estas pruebas. La
actualización de configuración requiere reconstruir las dependencias; no se
implementa un control de consentimiento en caliente en este frente.

Verificación del commit `a1d4c62`: **3598 passed, 9 failed, 1 xfailed** en
104,60 s; los nueve IDs de fallo coinciden con el registro anterior. Pasan las
66 pruebas específicas, Ruff 0.15.1, el gate medio/alto de Bandit 1.9.4, las seis
pruebas Node y el smoke autenticado del dashboard. El smoke del artefacto
compilado confirma además la denegación antes de red fuera del checkout. El
[registro de consentimiento](../research/2026-10-25-h1-consent-validation.json)
conserva alcance, comandos, resultados y comprobaciones reales pendientes.

## Política común y procedencia

`HomeActionService` conserva los recibos `HomeActionRequest`/`HomeActionResult`
y las lanes por entidad. Dispatcher, router legacy, HTTP, rutinas persistentes,
executor legacy y ambos caminos de entrega de recordatorios pasan por él.
También se protegió el executor de comandos aprendidos, aunque `main` no lo
construye actualmente. Los hooks opcionales pueden bloquear o reescribir;
el servicio valida siempre el resultado final. Los controles usan el intento
REST único ya existente y conservan la confirmación de estado fresco.

`HomeActor` separa la sesión del dashboard del actor doméstico compartido sin
identidad verificada. `ActionOrigin` registra HTTP, voz, rutina o recordatorio.
El cuerpo HTTP no admite actor ni origen; la sesión los aporta. SpeakerID,
nombres sugeridos y el `actor` guardado como auditoría en una rutina no otorgan
autenticación a un job. La identidad del recibo incluye el tipo de actor.

`HomeControlPolicy` requiere selección explícita de entidad, operación y rango.
`home_assistant.control_selection_path` apunta a datos privados de revisión,
fuera de Git; vacío deniega controles. El lector exige `selection_approved`,
`live_api_verified`, `allowed_entities[].live_verified` e
`indirect_effects_verified` booleanos verdaderos. Este último exige revisión de
las automatizaciones/bindings relevantes; no se deduce de capacidades ni estado.
La entrada declara `entity_id`, `operations` y, para brillo,
`parameters.brightness_pct` con `min` y `max` enteros. No hay permiso por dominio.
La selección se lee sin bloquear el loop y se vuelve a comprobar tras esperar
la lane y consultar HA. Archivo ausente, inválido o revocado no reutiliza una
copia anterior. Su actualización operativa debe reemplazar el archivo completo.

El perfil versionado conserva el path vacío. La revisión privada recibida aún
produce cero grants efectivos: no se cambió su evidencia ni se habilitaron
dispositivos. Grupos y escenas requieren una definición revisada y expansión a hojas
autorizadas; los scripts y macros opacos siguen rechazados. La normalización de productores
rechaza parámetros desconocidos, selectores de múltiples targets y servicios
fuera de las operaciones tipadas actuales, sin descartarlos silenciosamente.

La admisión de rutinas y recordatorios comprueba la selección, y su ejecución
la revisa otra vez. Revocar permisos no impide deshabilitar una rutina. El
manager legacy dejó de instalar o eliminar automatizaciones propiedad de HA:
esas automatizaciones no podrían consultar la política de KZA al dispararse.
H5 sigue obligado a conectar creación y gestión por voz al scheduler persistente;
el rechazo provisional de esa vía antigua no completa esa función.

Los recordatorios agregan una tabla de recibos por ocurrencia y acción. El
cuerpo se guarda antes del efecto; un ID con otro cuerpo produce conflicto.
Después de un reinicio, un envío interrumpido queda incierto y no se reenvía.
Las entregas por vencimiento y llegada comparten ese claim. Una secuencia sólo
avanza tras una acción confirmada. La migración aditiva se prueba en una copia
offline sintética preservando sus filas; no se migró la base instalada.

`failed`, `uncertain` y `confirmed` conservan significados distintos: aceptación
HTTP de HA no prueba el efecto observado. La confirmación de luces usa estado
fresco y brillo, con el plazo existente. Los recibos inmediatos siguen siendo
locales al proceso; ausencia o expulsión de un recibo no autorizan un reintento.

La confirmación es específica por operación y está acotada por timeout. Las
acciones programadas conservan su procedencia e ID durable; no recuperan un
envío incierto reintentándolo sin evidencia. La política rechaza locks y alarmas
incluso desde una escena o una sesión del dashboard. Los nombres de grupos y
perfiles vocales nunca alteran ese resultado.

Los modos/rangos acordados siguen en la revisión privada del inventario,
junto con su evidencia pendiente. La demo de IDs infantiles por SpeakerID ya
no se importa al cargar las políticas predeterminadas; se conserva sólo como
instrumento de tests explícitos. El servicio común sigue siendo obligatorio.

## Operaciones ampliadas y formularios

`HomeOperation` comparte el vocabulario de operaciones con rutinas. Los DTOs
conservan `temperature`, `hvac_mode`, `position`, `color_temp_kelvin` y brillo
opcional al ajustar color. Los parsers rechazan campos de otra operación,
parámetros libres, booleanos numéricos, NaN e infinitos. No se habilitan fan,
preset, swing, tilt ni rangos dobles de temperatura como consecuencia de elegir
un dominio.

La consigna de clima usa límites domésticos seleccionados y capacidades frescas
del equipo. Se comprueban feature, mínimo, máximo y paso; no se recorta un valor
inválido. La unidad de instalación se consulta en `/api/config` antes de enviar y
durante la confirmación: debe ser °C. Se confirma `temperature` como setpoint y
el modo HVAC previo, sin confundirlo con temperatura ambiente ni aceptar un
cambio de modo no solicitado. Los modos se comprueban contra selección y
`hvac_modes` actuales. Referencias: [REST](https://developers.home-assistant.io/docs/api/rest/)
y [climate en HA 2026.4.3](https://github.com/home-assistant/core/blob/2026.4.3/homeassistant/components/climate/__init__.py).

Cover admite abrir, cerrar, detener y posición. Exige una clase de persiana
seleccionada (`blind`, `curtain`, `shade` o `shutter`) y los flags observados de
cada operación; no habilita puertas, portones o garajes. Posición se confirma
contra `current_position`. Abrir/cerrar exige el extremo si HA informa posición;
si no, sólo confirma el estado terminal informado. Detener necesita dos lecturas
sin movimiento y posición estable si está informada. El YAML candidato fija
60 s para recorrido/posición, con validación estricta de timeouts; los demás
controles conservan el plazo existente. Son límites por validar en H6, no tiempos
de equipo medidos. Agotarlos deja un resultado incierto sin reenvío. Referencia:
[cover en HA 2026.4.3](https://github.com/home-assistant/core/blob/2026.4.3/homeassistant/components/cover/__init__.py).

Temperatura de color exige soporte `color_temp` y rango Kelvin observado y
seleccionado. El envío conserva brillo cuando se pide junto al color; la
confirmación exige modo de color y Kelvin exactos, además de brillo cuando
corresponde. Un dispositivo que redondee sin reportar el objetivo exacto queda
incierto; la validación de efectos reales sigue pendiente. Referencia:
[light en HA 2026.4.3](https://github.com/home-assistant/core/blob/2026.4.3/homeassistant/components/light/__init__.py).

HTTP, el adaptador de servicios de los productores, persistencia de rutinas y
recibos de recordatorios conservan estos campos. Las pruebas del dispatcher con
órdenes normalizadas se complementan ahora con las de texto hablado descritas
abajo. No se conectó el clasificador experimental de clima que tenía evaluación
NO-GO. La selección privada real sigue sin grants efectivos; estos contratos se
prueban con entidades sintéticas.

### Interpretación y resolución desde voz

`parse_home_command` reconoce una sola orden completa de temperatura absoluta,
modo HVAC, apertura/cierre/parada/posición de persiana, o color Kelvin con brillo
opcional. Conserva decimales y exige una composición válida de números hablados:
«dos tres» no se suma como cinco. Los valores fuera de rango o paso se rechazan
en el servicio; no se recortan. «Prendé el aire» pide modo o temperatura, sin
adivinar un modo. Los pasados «dejé», «ajusté», «configuré», «activé»,
«desactivé» y «paré» se bloquean antes de normalizar acentos.

Ambos modos de `RequestRouter` usan el parser después del gate de captura y
semántica. El canal de wake textual, ambient estricto y wakes débiles conservan
un gate adicional de comando: error, indisponibilidad o confianza insuficiente
impiden actuar. Sus slots y targets no sustituyen los del parser. Una orden
ampliada incompleta o ambigua no cae a similitud vectorial ni a un LLM para
inventar parámetros.

`main` inyecta `HomeVoiceResolver` con el mismo servicio de acciones y los defaults
de habitación configurados. Una habitación explícita debe coincidir exactamente
con un ID, nombre, display name o alias configurado y tener el equipo de ese
dominio seleccionado. No se sustituye una habitación desconocida por la del
micrófono. Sin habitación explícita se usa el default del micrófono; sin ninguna
evidencia de habitación sólo se admite una única entidad seleccionada del
dominio. No hay búsqueda aproximada de dispositivos por su friendly name.

`VoiceHomeActionExecutor` comparte hooks, intento único, recibos y feedback entre
el dispatcher y ambos modos del router. Tras cualquier rewrite, el servicio
obligatorio revisa destino, operación, parámetros, selección actual y estado
fresco. Un recibo incierto no produce éxito ni reintento. Se retiró el precheck
legacy de permisos por SpeakerID; tests con perfiles ausentes, guest y admin
comprueban que no cambia la selección autorizada. Se conservan perfiles para
funciones de usuario, sin convertirlos en actores autenticados.

El corpus y los grafos conectados usan transcripciones y dispositivos sintéticos;
no prueban reconocimiento acústico, timings ni efectos de los dispositivos reales.
La expansión de grupos/escenas se describe abajo; las persianas de test no
implican seleccionar un equipo real.

El catálogo usa la misma selección que el servicio. Sus controles son la
intersección con capacidades y límites actuales; sin selección quedan de lectura.
La UI de Casa y el editor de rutinas muestran unidades, modos seleccionados y
límites. El recibo conserva los valores solicitados. El smoke manual
`python -m tests.frontend.browser_typed_controls` usa Chromium, HTTP autenticado,
el servicio real y SQLite temporal con HA simulado: elige consigna y color con
brillo, y guarda una rutina decimal sin ejecutarla. No usa HA ni audio reales.

Verificación del commit `3ef6beb`: **3927 passed, 5 failed, 1 xfailed** en
108,82 s, sin IDs de fallo nuevos. Pasan 902 pruebas dirigidas y 233 de regresión
del pipeline, incluidas 49 de parser y 150 conectadas desde texto hablado en
ambos modos. Pasan Ruff 0.15.1, el gate medio/alto de Bandit 1.9.4, los diez tests
Node, el smoke autenticado del dashboard y el artefacto compilado fuera del
checkout. El [registro de voz](../research/2026-10-25-h1-voice-commands-validation.json)
detalla el alcance sintético y las validaciones reales pendientes.

Verificación del commit `48c19f3`: **3726 passed, 5 failed, 1 xfailed** en
108,59 s, con los mismos cinco IDs pendientes. Pasan 562 pruebas enfocadas,
incluidas 68 de contratos conectados, diez tests Node, el smoke Chromium y el
smoke compilado de operaciones y rutinas. Ruff 0.15.1 y el gate medio/alto de
Bandit 1.9.4 pasan. El
[registro de operaciones](../research/2026-10-25-h1-typed-operations-validation.json)
conserva comandos, alcance y límites de esa evidencia.

Verificación del commit `d49fa2f`: **3651 passed, 5 failed, 1 xfailed** en
105,16 s, sin IDs de fallo nuevos. Pasan 516 pruebas enfocadas, incluidas 48 de
grafos de política, y el smoke compilado que deniega controles sin selección
antes de usar HA. Ruff 0.15.1, el gate medio/alto de Bandit 1.9.4, las seis
pruebas Node y el smoke autenticado del dashboard pasan. El
[registro de política](../research/2026-10-25-h1-policy-validation.json) describe
las pruebas y sus límites.

## Grupos y escenas con efectos explícitos

`EffectPlan` expande los grupos de iluminación seleccionados y las escenas de
iluminación revisadas a `EffectIntent` de hojas concretas. No envía servicios de
HA a grupos, escenas, scripts o automatizaciones. La copia revisada de una escena
sólo acepta `state` on/off, kelvin y brillo 0–255; rechaza otros atributos,
templates, targets sensibles y valores contradictorios. Conserva el brillo
original: convertir 179 a un porcentaje entero podría modificar la consigna.
Los contratos, política, persistencia y formularios conservan `brightness` crudo
como alternativa exclusiva a `brightness_pct`, con límites porcentuales
comprobados sin redondear y confirmación exacta del atributo crudo.

Esta elección se basa en las implementaciones de HA 2026.4.3:
[grupos de luces](https://github.com/home-assistant/core/blob/2026.4.3/homeassistant/components/group/light.py)
reenvía a sus miembros; [escenas](https://github.com/home-assistant/core/blob/2026.4.3/homeassistant/components/homeassistant/scene.py)
reproduce estados por dominio, y [reproducción de luz](https://github.com/home-assistant/core/blob/2026.4.3/homeassistant/components/light/reproduce_state.py)
utiliza brillo crudo y atributos de color. El estado agregado no confirma cada
hoja y una invocación opaca no es evidencia de efectos permitidos.

Cada grupo de la revisión privada declara `kind: light_group`, `members`,
`membership_source` y `effects_verified: true`, además de operaciones y límites.
`ha_state` exige que la lista de miembros de HA coincida antes de los envíos.
`reviewed_copy` permite fijar una lista explícita cuando el bridge no publica
miembros en HA; se comprueba la existencia/disponibilidad de la referencia y,
si HA publica miembros, deben coincidir. KZA nunca llama al servicio del bridge:
cambios externos no agregan miembros a la copia autorizada. No es sincronización
automática con la configuración MQTT; revisar la copia es responsabilidad del
entorno de release y requiere nueva evidencia ante cambios externos.

Las escenas declaran `kind: scene`, nombre/aliases, `states`, operación
`activate_scene` y `effects_verified: true`. Sus referencias HA deben coincidir,
pero los valores ejecutados son los de la copia revisada, no una definición HA
que pueda cambiar y agregar efectos ocultos. Scripts, otras escenas anidadas y
otros dominios dentro de estas escenas de iluminación se rechazan. La revisión
de efectos indirectos es una declaración del operador respaldada por H0/H6;
no hay un detector automático de cambios en toda la configuración HA/MQTT.

Los miembros y nodos internos tienen grants con operaciones/rangos explícitos.
`direct_control: false` los autoriza sólo como parte de una expansión revisada:
no aparecen como controles nuevos en Casa ni son candidatos independientes de
voz. Una llamada directa o la creación de una acción diferida sobre ese miembro
se rechaza. Elegir un grupo no concede permisos generales sobre sus miembros.

La expansión tiene límites de ocho niveles, 128 hojas y 512 referencias
recorridas. Ciclos, membresía duplicada y solapamientos con consignas distintas
se rechazan; una hoja repetida con intención idéntica se ejecuta una sola vez.
Las lanes de todas las entidades se adquieren en orden estable, compartidas con
controles directos, para serializar escenas solapadas sin deadlocks. Se comprueba
todo el plan antes del primer envío y se vuelve a revisar selección, definición,
membresía disponible y capacidades frescas antes de cada hoja. No hay rollback
físico ni atomicidad frente a otros clientes de HA.

El recibo del conjunto conserva intención y resultado de cada hoja, incluyendo
`not_executed`. Un fallo antes de enviar no cambia ninguna hoja. Si hubo efectos
confirmados y luego falla otra parte, el conjunto queda incierto con
`partial_effects`; no avanza ni reintenta. La UI muestra cuántos dispositivos se
confirmaron, quedaron sin confirmar o no se ejecutaron. Los recibos detallados
del servicio siguen siendo locales al proceso; los jobs conservan su claim y
resultado raíz durables, que impiden repetir una escena parcial tras reinicio.
H5 debe conservar ese contrato al completar sus flujos de voz.

HTTP autenticado, ambos routers de voz, dispatcher, rutinas y recordatorios usan
el mismo servicio. El catálogo intersecta capacidades y límites de todas las
hojas. La voz exige el nombre/alias exacto de una escena seleccionada, sin
inventar un destino por similitud. El gate compartido también protege escenas
en pasado, negadas o diferidas. El smoke Chromium activa la escena sintética,
observa brillo crudo, muestra un grupo parcial y guarda una rutina de temperatura
decimal más escena sin ejecutarla.

Se actualizan cuatro tests legacy que todavía exigían fallback global al no
resolver una habitación. Sus nuevas aserciones requieren `None`, una sola query
y el filtro de área solicitado; se conserva el caso global sin área explícita
y el test independiente de exclusiones. No se modifica el runtime para volver
a accionar otra habitación. Queda pendiente el fallo VAD/idioma de H3.

El borrador privado de efectos conserva las siete escenas elegidas y los enlaces
de grupos hallados en el snapshot. Los grupos HA apuntan a grupos MQTT sin
miembros físicos en ese inventario: esa capa sigue sin completar y sin grants
efectivos. `live_api_verified` sigue falso y el YAML conserva el path de selección
vacío. Se solicitó el export privado de miembros para contrastarlo; ningún flag
se activa a partir de esta implementación sintética.

Verificación del commit `9ebdab8`: **4002 passed, 1 failed, 1 xfailed** en
111,08 s. El único fallo restante es idioma/VAD; no hay IDs de fallo nuevos.
Las 68 pruebas conectadas de expansión cubren los casos anteriores. Pasan
Ruff 0.15.1, el gate medio/alto de Bandit 1.9.4, los doce tests Node, el smoke
autenticado del dashboard y el artefacto compilado fuera del checkout. Chromium
comprueba seis envíos sobre hojas sintéticas, feedback parcial y guardado de una
rutina con consigna decimal más escena, sin ejecutarla; no hay excepciones JS.
El [registro de expansión](../research/2026-10-25-h1-effect-expansion-validation.json)
conserva comandos, fallos cerrados, límites y evidencia real pendiente.

## Cierre

H1 requiere todos los frentes de la tabla y pruebas conectadas para cada
productor, con HTTP/WS autenticados, política obligatoria y consentimiento
aplicado. La nueva frontera cierra cuatro fallos del dispatcher mediante fixtures
con selección explícita, estado fresco e intento único, conservando confirmación.
Los cuatro contratos obsoletos de búsqueda por habitación se actualizan para
comprobar ausencia de fallback; el fallo de idioma/VAD corresponde a H3. H1 aún
necesita la evidencia de miembros físicos, efectos indirectos, configuración y
comportamiento del hogar real. Las pruebas sintéticas no cierran esos puntos.
No hay publicación intermedia.
