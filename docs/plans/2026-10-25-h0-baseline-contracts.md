# H0: baseline, contratos y ensayo de release

Este documento ejecuta H0 del [roadmap](2026-10-25-stabilization-roadmap.md).
La referencia inicial es `f56b5ae`, en `recovery/production-2026-10-25`.
El trabajo se hace como `yo`, desde `/home/yo/projects/kza`. No modifica
servicios, permisos ni datos de producción. Se recibió el inventario privado de
HA y se revisó la selección inicial con el usuario. La metadata legacy también
se inventarió por lectura y tiene vista previa privada; H0 permanece abierto
hasta resolver su tratamiento y la evidencia faltante del inventario.

## Baseline de los diez fallos

Se reprodujeron los diez fallos, con 70 tests vecinos pasando, en CPython
3.13.12, sin GPU. Se usó el intérprete existente de producción **como `yo`**, en
modo de lectura y sin instalar dependencias. Las versiones no son los pins de
CI: esto clasifica el rescate, no valida dependencias reproducibles para H6.

Comando de reproducción del conjunto que contiene los diez fallos:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/kza/kza/.venv/bin/python -m pytest -q --tb=long \
  tests/unit/ambient/test_language_quality.py \
  tests/unit/dashboard/test_admin.py \
  tests/unit/orchestrator/test_dispatcher_ha_failure_feedback.py \
  tests/unit/orchestrator/test_dispatcher_hooks.py \
  tests/unit/orchestrator/test_dispatcher_unavailable_precheck.py \
  tests/unit/vectordb/test_search_command_excluded.py \
  tests/unit/vectordb/test_search_command_prefer_area.py
```

Los nombres completos originales están en el
[manifiesto del rescate](../research/2026-10-25-production-recovery.json).
No se marcan como xfail ni se cambian sus expectativas durante esta clasificación.

| Test (archivo indicado arriba) | Causa verificada en el código y el traceback | Clasificación y prueba responsable del cierre |
|---|---|---|
| `test_drops_low_vad` | `is_spanish_keepable` ignora VAD expresamente; `lang_ok` mide idioma. La energía se filtra en el distiller. | Contrato anterior. H3 lo reemplaza por conservación del texto de baja energía y prueba conectada del filtrado de VAD en el consumidor, sin restaurar descarte de texto por energía. Ver `tests/integration/test_capture_retention.py`. |
| `test_no_token_configured_means_open_access` | `require_admin` delega en `require_dashboard_session`; sin token devuelve 503 y no crea el usuario. | Expectativa insegura obsoleta. H1 reemplaza el caso por rechazo y ausencia de efectos en todos los endpoints privados. |
| `TestEarconOnFailure::test_successful_call_stays_silent` | El mock sólo hace async `call_service_ws`; el nuevo `await confirm_entity_state` recibe un `MagicMock` síncrono y genera el earcon de fallo. | Doble incompleto. H1 conserva la confirmación real; probar aceptación con estado confirmado, no confirmado y excepción. |
| `test_dispatch_emits_ha_action_dispatched_after_event` | Mismo `MagicMock` síncrono: el evento se emite con `success=False` y el error de `await`. | Doble incompleto. H1 verifica que el recibo y el evento describan el mismo resultado confirmado o incierto. |
| `test_unavailable_precheck_disabled_still_calls_ha` | Fixture con `RequestDispatcher.__new__` omite `_ha_entity_locks` del constructor. Falla antes del envío. | Fixture incompleta. H1 debe construir el dispatcher normalmente y verificar envío único y resultado; el kill switch no omite autorización. |
| `test_unknown_availability_fails_open_and_calls` | Misma omisión de `_ha_entity_locks`; además falta un doble async de confirmación. | Fixture incompleta. H1 prueba disponibilidad desconocida separada de permiso denegado. |
| `test_excluded_top_candidate_skipped_with_prefer_area` | Sólo hay candidatos excluidos o de Cocina para una petición en Escritorio. El rescate rechaza cambiar de habitación. | Contrato anterior. H1 conserva exclusión y ámbito; probar candidato válido dentro del área y rechazo cuando sólo hay candidatos fuera. |
| `test_area_below_threshold_falls_back_to_global` | El candidato del área está bajo el umbral. `search_command` devuelve `None` antes del pase global. | Contrato anterior. H1 prueba que un match débil no cambie el destino. |
| `test_area_pass_empty_falls_back_to_global` | No hay match en el área pedida; se rechaza explícitamente el fallback global. | Contrato anterior. H1 prueba rechazo sin llamadas HA. |
| `test_prefer_area_unknown_area_falls_back_to_baseline` | Una habitación desconocida tampoco autoriza una entidad en otra área. | Contrato anterior. H1 prueba abstención y pedido de aclaración, conservando órdenes válidas con destino verificable. |

Son seis expectativas anteriores y cuatro problemas de fixtures/dobles. Esta
clasificación no demuestra que sus caminos productivos estén libres de defectos:
el cierre exige las pruebas conectadas indicadas y la política compartida. Los
diez fallos permanecen abiertos en el baseline hasta esos cambios.

La suite completa posterior a las herramientas de H0 produjo **3123 passed,
10 failed, 1 xfailed**, en 79,69 s, con los mismos diez IDs de fallo. Los trece
tests nuevos de staging pasaron; `ruff check src/ tests/` y los dos scripts
nuevos pasaron. El [registro verificable](../research/2026-10-25-h0-baseline.json)
conserva comando, entorno, resultados y límites de esta validación.

## Matriz del grafo actual

Esta matriz distingue construcción de disponibilidad real. Se contrastaron
`main._run_main`, las factories y los consumidores; no se arrancó el asistente
ni se abrió un micrófono para producirla.

| Flujo | Construcción y consumo actuales | Diferencia que debe resolver el roadmap |
|---|---|---|
| HTTP de Casa y rutinas persistentes | `build_home_services` crea un `HomeActionService`; `_initialize_routines` lo pasa a `build_routine_scheduler` y al executor. Se construye con dashboard apagado. | H1: inventario aprobado, tipos clima/cover y política común en todos los productores. |
| Voz orquestada | `MultiUserOrchestrator` crea `RequestDispatcher`; `_fire_and_reconcile_ha` llama a `call_service_ws` y confirma luces/switches. Hooks opcionales pueden bloquear o reescribir. | H1: no comparte aún el servicio de Casa; autorización posterior a cualquier rewrite, aun sin hooks. |
| Voz legacy | `RequestRouter` recibe directamente HA y conserva un envío propio en `_process_command_legacy`. | H1: cerrar también este envío y probar ambos modos. |
| Escenas/automatizaciones | El catálogo las puede observar; el executor legacy conserva `ha_scene` y `ha_service`. | H1: membresía observada no verifica efectos. Rechazar definiciones opacas o expansiones restringidas. |
| Rutinas por voz | `RoutineManager(ha_client, chroma, llm)` es una implementación distinta del scheduler persistente entregado al dashboard. | H5: reemplazar el consumidor de voz por el mismo store/scheduler; no duplicar fuente de verdad. |
| Listas y recordatorios | Stores y managers se inicializan; el dispatcher recibe managers. `ReminderScheduler` se crea sólo con rooms habilitado y puede recibir presencia `None`. | H1 cierra acciones arbitrarias; H5 corrige presencia, creación por voz, zona y resultado anunciado. |
| Timers | `NamedTimerManager` se construye en `FeatureManager`, después de construir `RequestRouter`. El router tiene consumidor `self.features.timer_manager`. | H5: inyectar la misma instancia y probar el grafo real. |
| Alertas locales | `AlertManager` y `AlertScheduler` se construyen según config; faltan productores security/pattern/device en ese grafo. | H5: conectar sólo fuentes HA existentes e inventariadas; una alerta no concede control. |
| Reasoner | `HttpReasoner` se carga al iniciar; si falla queda ausente. El worker itera el stream síncrono. | H2: no bloquear loop, locución única, cancelación y backpressure. H4: reconexión. |
| Consentimiento | El gate se evalúa para reasoner y compactor; `is_cloud_endpoint` considera cualquier loopback local. | H1: declarar destino cloud independientemente del transporte loopback y probar todos los consumidores. |
| Ambient y memoria | `build_ambient_path` se conecta a multi-room; distiller usa `store_fact` con `origin=ambient`. Falta vínculo entre hecho e IDs fuente. | H3: TTL al leer, procedencia, cancelación de jobs borrados y migración con vista previa. |
| Voces | `SpeakerTagger` usa perfiles del identificador; sin perfiles devuelve `unknown`. No existe agrupador de desconocidos ni interfaz de revisión. | H3: IDs de grupo independientes de perfiles, correcciones durables, abstención y evaluación acústica. |
| Captura y diagnóstico | Multi-room integra monitorización y publica audio health al agregador/dashboard. El arranque requiere HA disponible. | H4: verificar recuperación de captura sin frames y dependencias tardías; H6 prueba unit real. |

Las pruebas actuales de cada módulo no bastan para afirmar que el grafo completo
funciona. H1–H5 deben construir el grafo con dobles en las fronteras de hardware y
red, y hacer que una petición recorra productores, autorización, envío y recibo.

## Contrato de acciones para H1

Se extenderán los DTOs y servicios existentes, manteniendo los recibos de Casa
y las rutinas persistentes. Los siguientes invariantes fijan sus interfaces:

- El actor es evidencia de autenticación, con tipo y alcance explícitos.
  `dashboard-admin` proviene de la sesión validada por el servidor. Una orden
  de voz tiene actor doméstico sin identidad verificada; no recibe el ID del
  usuario sugerido por SpeakerID como principal autenticado.
- El perfil de hablante, el grupo acústico y la sugerencia de nombre son
  anotaciones de procedencia. Ninguno amplía permisos, ni siquiera si una
  persona confirma manualmente el nombre en el dashboard.
- La intención tipada contiene `request_id`, origen (voz/HTTP/rutina/recordatorio),
  entidad, operación y parámetros normalizados. No contiene un servicio HA
  arbitrario producido por el LLM. El actor se inyecta desde la frontera, no
  desde el cuerpo HTTP ni el texto del usuario.
- Una política compartida valida entidad seleccionada, operación, capacidades
  observadas y límites acordados. Se aplica al crear y otra vez al ejecutar
  acciones diferidas; revisar una rutina no congela permisos para siempre.
  Locks y alarmas se rechazan para todo actor del producto.
- Clima y cover requieren tipos, confirmación y límites concretos antes de
  habilitarse. No se infieren permisos de un dominio ni de un nombre amigable.
  Una escena debe poder expandirse a acciones verificadas, con límites sobre
  anidamiento y rechazo de efectos opacos; la llamada `scene.turn_on` por sí
  sola no prueba esos efectos.
- Los hooks pueden restringir o reescribir; el resultado final vuelve a pasar
  la política obligatoria. Apagar hooks no apaga la política.
- Un ID reutilizado con otro cuerpo produce conflicto. El mismo actor e ID
  comparten envío/recibo; no se reintenta una acción incierta a ciegas.
  Se conservan `failed`, `uncertain` y `confirmed`: éxito HTTP de HA sólo
  confirma aceptación de transporte. La confirmación observa el estado
  apropiado a cada operación con plazo acotado.
- Las rutinas/recordatorios creados por voz desconocida son domésticos y
  carecen de titular personal. Su alcance será el mismo de las acciones
  inmediatas seleccionadas; los borrados amplios y la administración siguen
  en sesión autenticada. H5 debe concretar selección y cancelación sin
  inventar propiedad a partir de una huella vocal.

Pruebas de H1: matrices por productor y operación, ausencia de llamadas HA en
rechazos, escena con efecto restringido, permisos retirados entre programación
y ejecución, rewrite malicioso, hooks apagados y recibo incierto tras pérdida
de respuesta. Se añade corpus gate → router → envío para pasado, preguntas,
negaciones, condiciones y diferidos.

## Contrato de privacidad y voces para H3

Cada captura tendrá ID estable, tiempo de origen, conversación y procedencia.
Texto y clip vencen a las 48 h desde la captura, también al leer y tras reinicio.
Los hechos automáticos guardan IDs fuente y tiempo de extracción, vencen a
los 30 días y no renuevan TTL al consultar. Confirmar un recuerdo requiere
sesión del dashboard y deja constancia separada de la propuesta por voz.

El borrado de conversación deja una marca durable que consultan los jobs antes
de escribir; elimina clips, texto, hechos y vínculos derivados. El plan de H3
concretará la transacción/reconciliación entre SQLite y Chroma y cómo reintenta
una eliminación interrumpida sin volver a crear recuerdos.

El grupo acústico tiene ID propio y conserva evidencia y revisión manual por
separado. Las sugerencias pueden abstenerse; `unknown` no es un grupo de una
persona. Nombrar, unir o separar no modifica el actor de autorización ni la
clasificación de voz viva. Pausa ambiental y mute total son estados distintos;
el mute requiere reactivación por control que no dependa de audio.

Antes de modificar datos legacy se necesita una copia privada y una vista
previa: cantidad por colección/esquema, metadatos disponibles, registros sin
tiempo de origen, confirmación o fuentes, tratamiento propuesto y compatibilidad
de rollback. `created_at` de inserción y `origin=ambient` no prueban autor,
confirmación ni conversación de origen. No se inventan esos campos ni se
aplica un borrado masivo implícito. No se abrió ni modificó la memoria de
producción durante la primera clasificación de H0. Después se pudo consultar
la metadata de la colección existente mediante SQLite `mode=ro`,
`PRAGMA query_only=ON` y una transacción de lectura. La vista previa privada
bajo `data/` no copia documentos ni embeddings: registra presencia de campos,
procedencia declarada y el efecto hipotético de usar la inserción como inicio
de 30 días. No la presenta como fecha de origen de la conversación ni como
confirmación autenticada, y no autoriza eliminaciones. Falta decidir tratamiento
y probar la migración sobre una copia consistente con rollback compatible.

Decisión del usuario del 26 de octubre para H3: huellas y grupos sin confirmar
vencen a las 48 h; perfiles confirmados manualmente se conservan hasta borrado
manual. La confirmación deberá registrar procedencia y fecha; una sugerencia
automática o etiqueta legacy no la prueba. Clips y texto conservan
su TTL de 48 h. Falta implementar y probar esta política y su migración.
Antes de evaluar se fijarán umbrales, corpus de calibración separado y límites
de uniones/separaciones erróneas. El corpus incluirá distancias, habitaciones,
otras personas, TV, eco y superposición. Ningún test de UI cierra esa evaluación.

## Inventario privado y selección

Al revisar H0 no había `.env` ni variables HA exportadas para `yo`; el `.env`
de producción no era legible. Después el usuario proporcionó un snapshot
privado de registros, atributos persistidos y definiciones de HA. Se contrastaron
dispositivos, áreas, grupos y efectos declarados, y se guardaron una revisión y
la selección del usuario en el mismo directorio privado bajo `data/`, con modo
0600. Los IDs, nombres, capacidades y preferencias concretas no se suben a Git.

El snapshot declara `live_api_verified=false`: no incluye catálogo de servicios
en vivo ni prueba disponibilidad física. Incluye definiciones explícitas de
grupos y escenas de iluminación; los scripts con efectos indirectos y las
automatizaciones que dependen de contexto conservan efectos sin resolver.
También hay una automatización sin definición recuperada. No se autoriza ninguna
de esas entradas opacas desde KZA.

El usuario seleccionó los grupos y escenas de iluminación revisados y el
equipo de clima con modos y rango concretos, registrados sólo en el inventario
privado. La selección se refiere a la futura release, tras verificar capacidades
y efectos. Para reproducir las escenas hacen falta temperatura de color tipada,
normalización de brillo y expansión explícita de grupos: no basta el conjunto
actual turn_on/turn_off/brightness ni se pueden descartar parámetros desconocidos.

La descripción del interruptor de iluminación coincide con un dispositivo de
entrada y una automatización en el snapshot; se conserva como asociación
propuesta, no como salida `switch` identificada. No hay `cover` observado ni
seleccionado todavía: se mantiene la implementación tipada dentro del alcance,
sin sustituirlo por relés de carga desconocida.

No se solicitan privilegios administrativos para desarrollar. Para completar la
evidencia en vivo se pueden usar credenciales de lectura en el entorno de
desarrollo o un snapshot de `/api/states` y `/api/services`. Esta herramienta
adicional consume esas listas; el snapshot persistido recibido tiene un esquema
distinto y se revisó directamente conservando su procedencia:

```bash
# Ambas salidas son privadas, dentro de data/ (ignorado por Git).
/home/kza/kza/.venv/bin/python scripts/inventory_home.py \
  --env-file .env data/h0-home-inventory.json
# Alternativa sin conexión al servidor:
/home/kza/kza/.venv/bin/python scripts/inventory_home.py \
  --snapshot data/ha-snapshot.json data/h0-home-inventory.json
```

La herramienta sólo hace GET a estados/servicios, no sigue redirects, no envía
acciones y crea el archivo nuevo con modo 0600. No imprime entidades ni
credenciales. Toda selección empieza en `approved=false`; escenas, scripts y
automatizaciones tienen `effects.verified=false`, incluso si listan miembros.
Esta captura adicional no reemplaza los registros/definiciones recibidos. Se
deben reconciliar sus observaciones con el inventario privado, sin promover una
capacidad histórica a permiso. Las fuentes útiles y los umbrales de alertas
locales quedan para el plan de H5. La selección no habilita acciones ahora.

## Instalación, promoción y rollback en staging

`scripts/stage_runtime.py` consume el artefacto de `build_runtime.py`. Comprueba
formato CPython 3.13, magic, SHA de cada archivo, conjunto de archivos y rutas;
rechaza fuentes `.py`, symlinks de archivo y entradas que escapen del destino.
El manifiesto permite verificar integridad; la autenticidad se contrasta con el
SHA del artefacto calculado bajo `yo` para el commit candidato.

Cada slot contiene `runtime/` y una copia privada `state/` de config, datos y
`.env` opcional. El runtime apunta sólo al estado de su slot. `current` se
reemplaza atómicamente después de verificarlo; activar el slot anterior recupera
**su runtime y su estado**, incluyendo el esquema de datos anterior. Un lock
serializa las operaciones de esta herramienta, no los escritores del asistente.

`--offline` exige declarar que todos los escritores están detenidos. No los
detiene ni comprueba por sí solo; no debe usarse sobre un estado vivo. SQLite,
Chroma, jobs nocturnos, indexadores y otros escritores deben quedar quietos al
tomar el respaldo y cambiar el puntero. El ensayo automatizado usa SQLite cerrada
y estado sintético; no copia datos reales ni toca producción.

```bash
# Ejemplo bajo yo; seed/config y seed/data son un respaldo offline o datos de prueba.
python3.13 scripts/build_runtime.py dist/h0-runtime.tar.gz
sha256sum dist/h0-runtime.tar.gz
python3.13 scripts/stage_runtime.py --root "$PWD/dist/h0-stage" --offline \
  prepare dist/h0-runtime.tar.gz previous --state-source "$PWD/data/h0-seed"
python3.13 scripts/stage_runtime.py --root "$PWD/dist/h0-stage" --offline activate previous
# Copiar el estado detenido del anterior a un slot nuevo antes de migrarlo.
python3.13 scripts/stage_runtime.py --root "$PWD/dist/h0-stage" --offline \
  prepare dist/h0-runtime.tar.gz candidate --state-source "$PWD/dist/h0-stage/slots/previous/state"
# Ejecutar migraciones y smoke tests sobre candidate; sólo luego promover.
python3.13 scripts/stage_runtime.py --root "$PWD/dist/h0-stage" --offline activate candidate
# Rollback, con todos los escritores detenidos de nuevo:
python3.13 scripts/stage_runtime.py --root "$PWD/dist/h0-stage" --offline activate previous
```

La suite `tests/unit/tools/test_stage_runtime.py` prueba dos commits distintos,
una migración SQLite de datos y esquema, cambio de configuración y restauración
de ambos al volver al slot previo. También prueba el import de `src.main.pyc`
y los cinco recursos del frontend desde otro directorio, en un proceso limpio,
sin abrir el pipeline. Rechaza candidatos corruptos sin cambiar `current`,
fuentes inesperados, archivos duplicados, traversal y estado redirigido.

H6 debe agregar dependencias reproducibles y recursos de modelos identificados
por versión/hash a la combinación compatible. Esta herramienta no instala
dependencias ni conecta modelos/logs, no ejecuta migraciones y no administra
systemd. El perfil candidato deberá resolver rutas absolutas, enlaces revisados,
invocaciones de scripts y units reales según
[DEPLOYMENT.md](../architecture/DEPLOYMENT.md). La copia genérica de estado
rechaza symlinks y archivos especiales para exigir revisión explícita.

El rollback restaura el punto previo a la migración: los cambios de datos
posteriores a la promoción quedan en el slot candidato para reconciliación
explícita. No se mezclan automáticamente esquemas incompatibles ni se promete
conservar escrituras posteriores al respaldo. No retirar ningún slot o respaldo
hasta validar la release y su recuperación. La intervención administrativa y
la prueba con audio, HA y unit reales corresponden a la única release de H6.

## Salida de H0

- Clasificación de los diez fallos: registrada; baseline aún con fallos abiertos.
- Matriz de construcción y contratos de actores/acciones/procedencia: registrada.
- Ensayo de artefacto, promoción y restauración conjunta de datos: automatizado.
- Inventario privado y selección inicial: recibidos y revisados; faltan servicios
  y disponibilidad en vivo, resolución completa de efectos y selección de alertas.
- Metadata de memoria legacy: inventariada por lectura, con vista previa privada
  de retención hipotética. Tratamiento y prueba de migración en copia: pendientes.

H1 puede desarrollar política y pruebas con inventarios sintéticos sin habilitar
entidades reales. H3 puede concretar interfaces; la migración de registros reales
espera la vista previa y la decisión de tratamiento. Los gates de H6 siguen
pendientes: este ensayo no constituye una release validada.
