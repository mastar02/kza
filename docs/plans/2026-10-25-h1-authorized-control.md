# H1: control autorizado y semántica segura

H1 implementa R01, R02, R03 y consentimiento de R08 del
[roadmap](2026-10-25-stabilization-roadmap.md). Parte del
[baseline y contratos de H0](2026-10-25-h0-baseline-contracts.md).
H0 todavía necesita evidencia viva y vista previa de memoria; se avanza con
contratos y pruebas sintéticas sin habilitar dispositivos ni modificar producción.

## Secuencia y criterios

| Frente | Trabajo y evidencia de cierre | Estado |
|---|---|---|
| Semántica | Corpus de pasado, estado, preguntas informativas, negación, condición, reporte y diferidos. Gate, gramática, evidencia del LLM y routers conectados deben rechazar; los pedidos inmediatos conservan dirección e infinitivos/cortesía válidos. | Primera implementación y regresiones sobre texto completo. |
| Captura parcial | Verificar early-dispatch y pretranscripciones: un prefijo imperativo no autoriza ejecutar antes de recibir «mañana», «no» u otra condición del mismo pedido. La configuración recuperada mantiene early-dispatch apagado. | Pendiente; impide cerrar R03 para todas las configuraciones. |
| Sesión | HTTP y WebSocket privados comparten autenticación cerrada sin token, incluyendo listas, recordatorios, presencia, observabilidad y `/ws/live`. Orígenes inválidos se rechazan con sesión válida. Login y recursos estáticos tienen excepciones explícitas. Bind loopback detrás del proxy y token vacío en `.env.example`. | Pendiente. |
| Política | Un servicio de acciones y un actor de procedencia por productor. Política obligatoria después de rewrites y antes de todo envío; permisos revisados también al disparar acciones diferidas. Tests sin hooks, con rewrite y revocación. | Pendiente. |
| Operaciones | Extender DTOs, validación y confirmación a clima y cover. Agregar temperatura de color para reproducir las escenas seleccionadas sin perder parámetros. Usar rangos, modos y entidades del inventario privado aprobado. | Pendiente. |
| Grupos y escenas | Expandir definiciones verificadas a hojas autorizadas y operaciones tipadas. Rechazar macros opacas, templates sin contexto y efectos indirectos no verificados. Pruebas de escena que oculta lock/alarm, grupos no autorizados y parámetros desconocidos. | Pendiente. |
| Consentimiento | Destino cloud explícito, independiente de host/transporte. Denegación probada en reasoner, compactor y cada consumidor; loopback no elude consentimiento. | Pendiente. |

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

Los tests cubren texto completo entregado al router. No prueban STT acústico,
FAR/FRR ni que el cierre de captura haya recibido toda la frase. El early worker
actual puede despachar un comando parcial antes de VAD y luego salir; además el
router prefiere `wake_text` cuando existe. H1 debe resolver la frescura y la
finalidad de esos textos antes de declarar R03 cerrado, sin usar una medición
de latencia para justificar que se ejecute un prefijo incompleto.

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
