# Pendientes de KZA después de contrastar el review

La prioridad es cerrar las vías de control sin autorización, evitar acciones que
el usuario no pidió y completar la respuesta de voz. El review aporta problemas
concretos para esos frentes; sus notas globales y las propuestas de funciones
nuevas no sirven como criterios de cierre.

Estado del código contrastado el 2026-10-25: rama
`recovery/production-2026-10-25`, revisión `2960e06`. Los pendientes siguientes
incluyen trabajo sin resolver y cierres parciales; no implican que nunca se haya
trabajado en esas áreas. La implementación y la publicación se siguen por
separado según [DEPLOYMENT.md](../architecture/DEPLOYMENT.md).

El [roadmap acordado](2026-10-25-stabilization-roadmap.md) define la secuencia y
el alcance de la única release. Incluye clima y persianas seleccionados,
retención de memoria y revisión de grupos de voz. Música y avisos externos de
sordera o caída quedan postergados; los pendientes R06 y R07 se ejecutan con ese
alcance. R11 se incorpora por un pedido posterior al review.

## Orden de trabajo

| ID | Prioridad | Pendiente | Estado actual |
|---|---|---|---|
| R01 | P0 | Cerrar acceso al dashboard privado | Protección parcial |
| R02 | P0 | Aplicar una política común a las acciones de HA | Pendiente de unificar |
| R03 | P0 | Evitar interpretar preguntas en pasado como órdenes | Bloqueos parciales |
| R04 | P1 | Completar el slow path sin bloquear la captura | Defectos confirmados |
| R05 | P1 | Reproducir todo el audio generado | Pérdida de samples confirmada |
| R06 | P1 | Recuperar dependencias y fallos de audio sin intervención | Arranque frágil y operación por validar |
| R07 | P1 | Completar las funciones de voz ya anunciadas | Cableado parcial |
| R08 | P1 | Hacer efectivas las opciones de privacidad | Controles parciales |
| R09 | P1 | Verificar el comportamiento del runtime recuperado | Suite con diez fallos |
| R10 | P2 | Alinear configuración y documentación con el runtime | Deriva confirmada |
| R11 | P1 | Revisar voces agrupadas por probable hablante | Nuevo alcance acordado |

R01 y R02 deben resolverse juntos para que cerrar un endpoint no deje otra vía
para ejecutar la misma acción. R03 es independiente. R09 acompaña cada arreglo;
su posición en la tabla no posterga las pruebas hasta el final.

## Criterios de cierre

### R01 Cerrar acceso al dashboard privado

El control directo de entidades, las rutinas nuevas, admin y `/ws` ya requieren
sesión. Siguen abiertos recordatorios, listas, presencia y rutas de observabilidad,
incluido `/ws/live`. El YAML mantiene `dashboard.host: 0.0.0.0`; el valor por
defecto de la clase es loopback.

Cerrar cuando las rutas privadas HTTP y WebSocket rechacen clientes sin sesión,
fallen cerradas sin token y apliquen la validación de origen correspondiente.
Documentar las excepciones públicas necesarias, el bind elegido y el acceso
mediante proxy si se adopta. Incluir el token en `.env.example` sin valor real.

Evidencia: [API](../../src/dashboard/api.py),
[autenticación](../../src/dashboard/auth.py),
[observabilidad](../../src/dashboard/observability.py).

### R02 Unificar la autorización de acciones

La allowlist de `HomeActionService` protege el control directo nuevo; no cubre
todas las llamadas desde voz y recordatorios. La API acepta recordatorios
anónimos con acciones arbitrarias y el scheduler las pasa al cliente HA. La
prueba aislada llegó a `lock.unlock` con un cliente simulado. Con presencia
deshabilitada, el cableado actual entrega `None` y falla antes de ese envío:
corregir presencia sin cerrar esta vía habilitaría el riesgo latente.

Cerrar cuando todos los productores de acciones apliquen una política explícita
de dominios y servicios, antes de enviar a HA. Probar rechazos desde HTTP, voz y
schedulers, incluidos usuarios desconocidos. Sustituir los IDs infantiles de
ejemplo por permisos reales si se conserva ese modo. La identidad por voz y una
confirmación verbal no sustituyen la autorización.

Evidencia: [acciones](../../src/home_assistant/home_actions.py),
[recordatorios](../../src/reminders/reminder_scheduler.py),
[política infantil](../../src/policies/permissions.py).

### R03 Completar los bloqueos semánticos

Las negaciones, condiciones y órdenes diferidas del review ya se rechazan.
`¿Apagaste la luz?` todavía pasa el gate y obtiene `turn_off` con confianza 0,85
cuando se confirma el wake acústico.

Cerrar con un corpus que distinga órdenes inmediatas, preguntas sobre acciones
pasadas, negaciones y pedidos diferidos. Probar el gate y el router conectados:
una frase rechazada no debe terminar ejecutándose por otra ruta. Conservar
pedidos válidos como `¿podés apagar la luz?`.

Evidencia: [gramática](../../src/nlu/command_grammar.py),
[gate](../../src/nlu/command_gate.py),
[router](../../src/pipeline/request_router.py).

### R04 Completar el slow path

El worker consume `generate_stream` de forma síncrona y el router excluye
`SLOW_LLM` de su locución. En la prueba aislada, un stream simulado de 200 ms
postergó hasta 211 ms una tarea que debía despertar a los 10 ms.

Cerrar cuando una consulta conversacional produzca voz, los errores tengan
respuesta audible y una generación lenta permita continuar la captura y atender
cancelaciones. Probar también caída del reasoner y timeout, sin llamadas cloud
reales en los tests de regresión.

Evidencia: [worker](../../src/orchestrator/dispatcher.py),
[respuesta](../../src/pipeline/request_router.py).

### R05 Evitar pérdida de audio

`StreamingAudioPlayer._write_to_buffer` trunca lo que no entra. Un chunk de
24.000 samples a 24 kHz conservó solo 3.600 samples, equivalentes a 150 ms.
Esto confirma el defecto de buffer; no establece una proporción fija de pérdida
para cualquier voz o respuesta.

Cerrar con conservación de todos los samples en reproducción normal, presión de
retorno al productor y cancelación sin bloqueos. Verificar chunks mayores que el
buffer y una respuesta larga no cacheada en el equipo. Barge-in y supresión de
eco requieren una prueba de audio real para dar por cerrado su funcionamiento.

Evidencia: [player](../../src/tts/piper_tts.py),
[reproducción por zona](../../src/audio/zone_manager.py).

### R06 Recuperar dependencias y vigilar el audio

El arranque sale si HA no responde, la unit versionada limita los reinicios y
un reasoner que falla al cargar queda en `None`. El watchdog de audio y el poller
ya tienen trabajo previo: falta verificar su operación efectiva y el diagnóstico
local, en lugar de asumir que no existen. Los avisos externos quedan fuera de
esta release por decisión del usuario.

Cerrar con una prueba de HA y gateway tardíos que se recuperen sin reiniciar KZA,
y una simulación de micrófono sin frames que produzca recuperación o un fallo
visible en el diagnóstico local. Revisar SIGTERM y cierre de recursos con la
unit real durante la validación de la próxima release. El preflight de VRAM
existente no equivale a recuperación ante OOM.

Evidencia: [arranque](../../src/main.py),
[unit](../../systemd/kza-voice.service),
[plan previo de fallos silenciosos](../superpowers/plans/2026-07-31-remediacion-fallos-silenciosos.md).

### R07 Completar las funciones de voz anunciadas

El scheduler persistente nuevo y su executor sí se construyen mediante la
factory. El manejo de rutinas por voz conserva otra implementación con llamadas
async sin `await`. Los timers no se inyectan al router; crear recordatorios por
voz conserva una respuesta de uso de API. Spotify no se construye desde el
entrypoint y las alertas carecen de sus productores en ese cableado.

Cerrar por función, con pruebas que recorran las dependencias construidas por
`main` y comprueben el resultado anunciado. Priorizar rutinas, timers y
recordatorios y conectar fuentes existentes de alertas locales con datos reales
de HA. Música y avisos externos de caída se evalúan en un roadmap posterior.
Reutilizar el scheduler nuevo; no reconstruirlo porque el review antiguo lo
consideraba desconectado.

Evidencia: [entrypoint](../../src/main.py),
[factory](../../src/routines/factory.py),
[rutinas por voz](../../src/routines/routine_manager.py),
[features](../../src/pipeline/feature_manager.py).

### R08 Hacer efectivas las opciones de privacidad

El distiller sí excluye fuentes `self` y `tv`, además de aplicar filtros de
calidad; admite `unknown`. El texto crudo tiene TTL, pero los hechos destilados
necesitan su propia política de conservación y borrado. El gate cloud permite
loopback aunque el gateway local pueda reenviar a un proveedor externo.

Cerrar con clasificación explícita del destino cloud, consentimiento aplicado a
todos sus consumidores y una prueba con gateway loopback. Definir y probar qué
se conserva, durante cuánto tiempo y cómo se pausa la captura y se elimina lo
derivado de una conversación. La eliminación debe alcanzar también los hechos
persistidos; filtrar TV no equivale a controlar el consentimiento de personas.

Evidencia: [consentimiento](../../src/llm/cloud_consent.py),
[selección ambiental](../../src/ambient/store.py),
[destilación](../../src/ambient/distiller.py).

### R09 Verificar el runtime recuperado

Medición de esta revisión con Python 3.13.12, sin GPU: 3.110 tests pasan, 10
fallan y 1 xfail; 84,32 s. Cobertura de líneas: 57,36 %, `main.py`: 17,39 %.
Se usaron dependencias existentes, distintas de los pins de CI. El manifiesto
del rescate registra los mismos diez fallos antes de agregar los tres tests del
builder. Uno de ellos todavía espera acceso administrativo sin token: hacerlo
pasar restaurando ese comportamiento sería una regresión.

Cerrar los diez fallos identificando cambios de contrato y defectos reales.
Agregar pruebas del código productivo para permisos y del grafo de construcción
para los flujos corregidos. Los 43 de 76 tests de safety que no importan `src`
no demuestran por sí solos la seguridad del runtime. Mejorar los mocks en las
interfaces que se toquen, en lugar de imponer una reescritura masiva.

Medir latencia de punta a punta y p95 con límites explícitos, y FAR/FRR del wake
con un corpus reproducible. Los contadores de wakes y los tiempos internos no
sustituyen esas métricas. El CI de `main` de agosto no valida esta rama.

Evidencia: [manifiesto del rescate](../research/2026-10-25-production-recovery.json),
[safety](../../tests/safety/), [CI](../../.github/workflows/ci.yml).

### R10 Alinear configuración y documentación

Corregir primero las claves consumidas de forma distinta a su estructura YAML,
incluida la configuración nocturna que omite GPUs y otros campos. Validar el
esquema de los subsistemas activos y documentar las claves inertes. Para el
entrenamiento nocturno, comprobar sus fuentes de datos y su utilidad antes de
mantenerlo habilitado; no dar por medido que el dataset siempre está vacío.

Cerrar con configuración validada, dependencias de ejecución reproducibles y una
matriz de capacidades que distinga implementación, cableado y validación en
equipo. Actualizar las afirmaciones de README, HARDWARE y CLAUDE sobre funciones
disponibles; CLAUDE ya describe dos GPUs y reasoner cloud. El mecanismo de
instalación, promoción y rollback del artefacto sigue el plan de despliegue
existente y debe completarse antes de publicar una release.

Evidencia: [configuración](../../config/settings.yaml),
[schema](../../src/core/settings_schema.py),
[entrenamiento](../../src/training/nightly_trainer.py),
[despliegue](../architecture/DEPLOYMENT.md).

### R11 Revisar voces agrupadas por probable hablante

El usuario pide ver grupos sugeridos de capturas que parecen corresponder a la
misma persona, asignarles nombres y corregir uniones o separaciones. La revisión
incluye transcripciones y fragmentos locales de audio con TTL de 48 horas.

El tagger actual compara con perfiles registrados; no agrupa hablantes
desconocidos. Cerrar con una vista autenticada, agrupación evaluada sobre audio
representativo, correcciones persistentes y un estado desconocido para casos
ambiguos. Definir la retención de huellas y perfiles por separado. La asociación
acústica o manual con un nombre no otorga permisos de control.

Evidencia: [tagger](../../src/ambient/speaker_tagger.py),
[identificador](../../src/users/speaker_identifier.py),
[almacenamiento](../../src/ambient/store.py).

## Trabajo existente que se conserva

Conservar la autenticación estricta y la allowlist ya incorporadas al control
directo, los bloqueos semánticos rescatados, el scheduler persistente con su
executor, el cableado de barge-in y los filtros ambientales. Sus pruebas y
extensiones pertenecen a los pendientes anteriores; no son módulos que haya que
crear otra vez.

El [backlog de marzo](2026-03-06-production-backlog.md) y los planes de
`superpowers/` aportan antecedentes. Sus estados históricos y checkboxes no
sustituyen los criterios de cierre de este backlog.

## Decisiones que requieren evidencia adicional

La nota de calidad 8/10 o 5/10, el porcentaje de código dormido y la cifra de
latencia de 210 ms no determinan tareas por sí solos. Una referencia estática
ausente en `main.py` tampoco prueba que una dependencia no se construya:
`RoutineExecutor` se instancia en la factory.

Eliminar Spotify, LoRA, learning o emoción; cambiar los micrófonos; y ampliar a
cámaras o energía son decisiones de producto posteriores. Música queda fuera de
esta release, sin comprometer ahora una nueva integración.

La exposición pública del repositorio merece revisar qué información operativa
se publica. Borrar documentos no revoca permisos ni borra su historia. Cambiar
la visibilidad, sudo o permisos de dispositivos instalados es una decisión
operativa separada: el acuerdo actual conserva `kza` como cuenta administrativa
y de ejecución, y desarrolla bajo `yo`. Ninguna tarea de este backlog adelanta
la retirada del checkout ni cambia producción fuera de una release.
