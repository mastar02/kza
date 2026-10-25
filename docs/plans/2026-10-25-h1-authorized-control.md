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
| Política | Un servicio de acciones y un actor de procedencia por productor. Política obligatoria después de rewrites y antes de todo envío; permisos revisados también al disparar acciones diferidas. Tests sin hooks, con rewrite y revocación. | Frontera compartida implementada; faltan integración del NLU y expansión verificada. |
| Operaciones | Extender DTOs, validación y confirmación a clima y cover. Agregar temperatura de color para reproducir las escenas seleccionadas sin perder parámetros. Usar rangos, modos y entidades del inventario privado aprobado. | Contratos, servicio, persistencia y formularios implementados. Falta interpretar y resolver estas órdenes desde el NLU de voz, e integrar expansión de efectos. |
| Grupos y escenas | Expandir definiciones verificadas a hojas autorizadas y operaciones tipadas. Rechazar macros opacas, templates sin contexto y efectos indirectos no verificados. Pruebas de escena que oculta lock/alarm, grupos no autorizados y parámetros desconocidos. | Pendiente. |
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
`live_api_verified` y `allowed_entities[].live_verified` booleanos verdaderos.
La entrada declara `entity_id`, `operations` y, para brillo,
`parameters.brightness_pct` con `min` y `max` enteros. No hay permiso por dominio.
La selección se lee sin bloquear el loop y se vuelve a comprobar tras esperar
la lane y consultar HA. Archivo ausente, inválido o revocado no reutiliza una
copia anterior. Su actualización operativa debe reemplazar el archivo completo.

El perfil versionado conserva el path vacío. La revisión privada recibida aún
produce cero grants efectivos: no se cambió su evidencia ni se habilitaron
dispositivos. Grupos observados, escenas y scripts opacos se rechazan hasta que
su expansión verificada tenga implementación. La normalización de productores
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
recibos de recordatorios conservan estos campos. Los tests del dispatcher parten
de órdenes ya normalizadas: todavía no prueban que una frase acústica llegue a
la entidad y parámetros correctos. No se conectó el clasificador experimental
de clima que tenía evaluación NO-GO. La selección privada real sigue sin grants
efectivos; estos contratos se prueban con entidades sintéticas.
Al integrar voz hay que retirar también el precheck legacy de `UserManager`
basado en el perfil SpeakerID; no debe bloquear el alcance doméstico seleccionado
ni convertir ese perfil en un principal autenticado.

El catálogo usa la misma selección que el servicio. Sus controles son la
intersección con capacidades y límites actuales; sin selección quedan de lectura.
La UI de Casa y el editor de rutinas muestran unidades, modos seleccionados y
límites. El recibo conserva los valores solicitados. El smoke manual
`python -m tests.frontend.browser_typed_controls` usa Chromium, HTTP autenticado,
el servicio real y SQLite temporal con HA simulado: elige consigna y color con
brillo, y guarda una rutina decimal sin ejecutarla. No usa HA ni audio reales.

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

## Cierre

H1 requiere todos los frentes de la tabla y pruebas conectadas para cada
productor, con HTTP/WS autenticados, política obligatoria y consentimiento
aplicado. La nueva frontera cierra cuatro fallos del dispatcher mediante fixtures
con selección explícita, estado fresco e intento único, conservando confirmación.
Siguen abiertos los cuatro fallos de búsqueda por habitación; no se restaurará
fallback a otra habitación para volverlos verdes. El quinto fallo, de idioma/VAD,
corresponde a H3. H1 aún necesita integrar las órdenes ampliadas desde el NLU de
voz y expandir los efectos verificados de grupos y escenas.
No hay publicación intermedia.
