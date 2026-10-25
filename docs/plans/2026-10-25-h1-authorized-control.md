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
| Política | Un servicio de acciones y un actor de procedencia por productor. Política obligatoria después de rewrites y antes de todo envío; permisos revisados también al disparar acciones diferidas. Tests sin hooks, con rewrite y revocación. | Pendiente. |
| Operaciones | Extender DTOs, validación y confirmación a clima y cover. Agregar temperatura de color para reproducir las escenas seleccionadas sin perder parámetros. Usar rangos, modos y entidades del inventario privado aprobado. | Pendiente. |
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

## Integración de la política pendiente

El plan conserva los recibos `HomeActionRequest`/`HomeActionResult`, las lanes
por entidad y la factory del scheduler persistente. Debe sustituir los envíos
propios del dispatcher, router legacy, scheduler de recordatorios y executor
legacy por esa frontera compartida. Ningún wrapper opcional de hooks puede ser
la única defensa. `failed`, `uncertain` y `confirmed` conservan significados
distintos; una respuesta aceptada de HA no prueba el efecto observado.

La confirmación es específica por operación y está acotada por timeout. Las
acciones programadas conservan su procedencia e ID durable; no recuperan un
envío incierto reintentándolo sin evidencia. La política rechaza locks y alarmas
incluso desde una escena o una sesión del dashboard. Los nombres de grupos y
perfiles vocales nunca alteran ese resultado.

Las capacidades se validarán contra observaciones actuales, y el esquema de
`config/settings.yaml` describirá los campos de la política y dónde leer su
selección privada. No se publicarán entidades personales en el YAML versionado
ni se agregará una allowlist implícita por dominio. Los modos/rangos acordados
siguen en la revisión privada del inventario, junto con su evidencia pendiente.

## Cierre

H1 requiere todos los frentes de la tabla y pruebas conectadas para cada
productor, con HTTP/WS autenticados, política obligatoria y consentimiento
aplicado. Las correcciones de fixtures y expectativas de los nueve fallos del
baseline asignados a H1 acompañan la nueva frontera, sin retirar confirmación de
estado ni restaurar acceso anónimo o fallback a otra habitación. El fallo de
idioma/VAD corresponde a H3. No hay publicación intermedia.
