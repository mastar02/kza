# Evaluación integral de KZA

KZA tiene una base técnica consistente y una suite amplia que pasa completa, pero varias tareas anunciadas todavía no terminan desde la voz hasta la persistencia y la entrega. La valoración orientativa del producto es **5,1/10**, con **7,5/10 para el núcleo técnico**. La prioridad es completar esas tareas, recuperar la captura de Cocina y hacer que la web muestre operaciones y observaciones verificables.

Observaciones del **1 de noviembre de 2026 en Uruguay**, correspondientes al **2 de noviembre UTC**. Checkout revisado: `32ca216`; runtime observado: `e9669ec`. El árbol de `src/` y `config/` no difiere entre ambos commits. Esta evaluación no modifica el despliegue ni acepta H6.

## Puntuación y criterio

La escala expresa juicio técnico, con confianza media: 3 indica un recorrido principal bloqueado, 5 una implementación parcial y 7 una base consistente con pruebas. Una nota de 10 exige resultados representativos de campo y tareas completas aceptadas. Los pesos conservan la línea base para permitir comparaciones.

| Área | Nota | Peso | Fundamento |
|---|---:|---:|---|
| Web Tu casa | 7/10 | 10 % | Se adapta al móvil, usa controles semánticos y comunica errores. Los recorridos locales de controles y memoria pasan. |
| Panel de observabilidad | 4/10 | 10 % | Revisión de voces conectada, junto a indicadores de ejemplo, controles sin operación y distribución móvil deficiente. |
| Usabilidad y accesibilidad | 5/10 | 15 % | Hay carga y errores explícitos, pero faltan navegación técnica por teclado, continuidad de foco y navegación recuperable. |
| Núcleo y arquitectura | 7,5/10 | 20 % | Contratos comunes, cierre explícito y 5.765 pruebas pasando. Persisten conexiones de funciones incompletas. |
| Captura y validación de audio | 4/10 | 15 % | Tres entradas reciben señal; falta Cocina y aceptación física, acústica y de reconocimiento. |
| Funciones de punta a punta | 3,5/10 | 15 % | Timers, recordatorios, rutinas y alertas tienen brechas concretas en el grafo de aplicación. |
| Operación e integraciones | 4/10 | 15 % | HA recuperó lectura; su WebSocket en KZA sigue desconectado y la salud de audio continúa degradada. |

La media ponderada es **5,1/10**. La variación respecto de la línea base de 5/10 refleja evidencia más completa y recuperación de acceso, no una mejora de código publicada. La calidad perceptual del audio queda sin nota.

## Estado operativo y dispositivos disponibles

Dos diagnósticos, a las 02:26 y 02:41 UTC, muestran voz y proxy activos sin cambio del proceso de voz. Living, Hall y Escritorio mantuvieron su generación de captura y sumaron aproximadamente 14 millones de frames cada uno. Cocina continúa desconectada y sin frames.

KZA ahora informa HA `healthy / connected`. Su contador de errores se mantuvo en 4.779 y los éxitos aumentaron de 1 a 4 entre observaciones; `ws_connected` sigue en `false`. Por separado, el token suministrado permite leer estados, servicios y los tres registros de áreas, entidades y dispositivos por WebSocket. Ese socket temporal de evaluación no demuestra recuperación de la suscripción de KZA.

| Observación de HA | Resultado | Alcance |
|---|---:|---|
| Estados publicados | 1.178 entidades | Son entidades, no dispositivos físicos distintos. |
| Estados informados | 197 | Estado distinto de `unknown` y `unavailable`; no comprueba exactitud ni efecto físico. |
| Estados no disponibles | 951 | Incluyen 436 rastreadores y 475 sensores; no equivale a 951 equipos averiados. |
| Estados desconocidos | 30 | Incluyen entidades cuyo estado no representa una lectura continua, como botones. |
| Registro de dispositivos | 484 | Inventario registrado, con entradas que pueden ser históricas. |
| Áreas | 11 | Sólo 43 de las 2.584 entidades del registro tienen área directa o heredada del dispositivo. |
| Luces | 8 de 9 con estado informado | Ocho tienen área asignada. |
| Interruptores | 24 de 37 con estado informado | 36 de los 37 no tienen área en HA. |
| Climatización | 1 con estado informado | Tiene área; operaciones y límites requieren selección de control. |
| Reproductores multimedia | 2 no disponibles | El inventario actual no respalda una experiencia musical operativa. |

Las asignaciones de HA deben conciliarse con las asignaciones explícitas de KZA. Ver una entidad o un servicio no concede permiso para controlarlo. El inventario obtenido no añadió controles autorizados ni ejecutó servicios.

Hay candidatos para alertas entre los sensores que informan estado: 2 de batería, 33 de energía, 15 de potencia y 48 sensores binarios de problema. Cada fuente necesita revisión de significado, utilidad y configuración antes de producir avisos.

El monitor conserva cuatro comandos históricos, todos fallidos, con p50 de 2.624,7 ms y p95 de 6.791,5 ms. La muestra es insuficiente para caracterizar latencia y no incluye una nueva prueba de voz. Las cifras históricas de 150–280 ms no quedan aceptadas por este diagnóstico.

## Matriz de capacidades

| Función | Estado de implementación | Qué falta para considerarla completa |
|---|---|---|
| Domótica | Voz y web comparten `HomeActionService`, resolución y política; hay pruebas de operaciones y recibos. | Selección de entidades, parámetros y efectos revisados; comprobación física de tareas desde audio. |
| Consultas y clima | Routing de consultas y `FAST_WEATHER` conectados a HA; existe ruta de pronóstico. | Calidad del razonamiento, presupuestos por recorrido y resultados reales con corpus. |
| Listas | Gestor persistente y CRUD del dispatcher conectados. | Comprobar frase, actor, persistencia y resultado; distinguir turno atendido de cambio realizado. |
| Timers | El gestor se construye dentro de `FeatureManager`. | `main` no pasa `features` a `RequestRouter`; crear, consultar, cancelar y avisar deben recorrer el grafo real. |
| Recordatorios | Store, manager, consultas y scheduler existen. | Crear por voz, cancelar el objeto correcto y evitar cerrar como entregado un aviso sin salida. |
| Rutinas | La web usa scheduler persistente y control común. | Voz conserva `RoutineManager` heredado, una llamada async sin `await` y rechazo explícito de la creación antigua. |
| Música | Hay subsistema y pruebas aisladas. | `main` no construye ni inyecta `MusicDispatcher`; queda en el roadmap posterior vigente. |
| Alertas | Manager, scheduler y salida común construidos. | Las tres listas de productores empiezan vacías y la fábrica no registra las fuentes automáticas. |
| Memoria | Procedencia, propuestas, revisión, caducidad y retirada conectadas a los recorridos. | Calidad semántica con corpus y continuidad de foco en la revisión web. |
| Voces y usuarios | Capturas, grupos sugeridos y revisión manual integrados. | Calibración acústica con voces consentidas; nombrar un grupo no autentica una persona ni concede permisos. |
| Intercom y notificaciones | Productores construidos y salida común disponible. | Los handlers de voz dependen de la misma conexión pendiente de `RequestRouter.features`. |

Fuentes del grafo: [main.py](../../src/main.py#L1507), [RequestRouter](../../src/pipeline/request_router.py#L1228), [FeatureManager](../../src/pipeline/feature_manager.py#L415), [dispatcher](../../src/orchestrator/dispatcher.py#L634) y [fábrica de servicios de voz](../../src/pipeline/voice_services.py). La evidencia detallada por capacidad está en el JSON adjunto.

## Defectos funcionales prioritarios

1. **Un recordatorio puede anunciar éxito sin crearse.** La rama de creación pide usar la API y devuelve `success=True`. La reproducción local confirma esa respuesta sin llamada al gestor de creación. Conviene separar `handled`, persistencia y entrega, de modo que las métricas representen tareas conseguidas. Fuente: [dispatcher.py](../../src/orchestrator/dispatcher.py#L1461).

2. **Cancelar un recordatorio cancela otra cosa.** Con servicios simulados, `dispatch("cancela el recordatorio de la basura")` devuelve intención `cancel`, llama una vez a la cancelación de cola y no llama a `cancel_by_text`. La cancelación global se procesa antes de clasificar el objeto explícito. Esta reproducción cubre el dispatcher, no los gates previos de audio. Fuente: [dispatcher.py](../../src/orchestrator/dispatcher.py#L688).

3. **Timers e intercom carecen de conexión al router de `main`.** El constructor omite `features`; guardarlo por separado en `VoicePipeline` no lo conecta. Una aceptación debe entrar por la fábrica real, no por una llamada manual al manager. Fuentes: [main.py](../../src/main.py#L1507) y [voice_pipeline.py](../../src/pipeline/voice_pipeline.py#L69).

4. **Las rutinas de voz y web tienen fuentes distintas.** La voz llama `get_domotics_entities()` sin `await` y trata el resultado como una lista; además, la escritura heredada está deliberadamente rechazada. La corrección debe usar el scheduler persistente y la política común, conservando ese rechazo del atajo antiguo. Fuente: [routine_manager.py](../../src/routines/routine_manager.py#L178).

5. **Un recordatorio sin zona puede finalizar sin voz.** Cuando existe detector de presencia, se agotan los reintentos y la zona sigue ausente, el scheduler puede omitir TTS y continuar hasta `fired` o reprogramación. Hace falta una política explícita de pendiente, aviso alternativo o perdido, con entrega observada. Sin detector ya existe fallback `output_zone`; ese pendiente antiguo no debe repetirse como defecto actual. Fuente: [reminder_scheduler.py](../../src/reminders/reminder_scheduler.py#L136).

6. **El scheduler de alertas activo no prueba detección automática.** Sus listas de handlers están vacías y la fábrica no las registra. Crear manualmente una alerta en un test verifica otro recorrido. Fuente: [alert_scheduler.py](../../src/alerts/alert_scheduler.py#L115).

7. **El watchdog espera recuperaciones lentas antes de volver a publicar.** El código mantiene `await _recover_streams(stale)` dentro del mismo bucle que publica heartbeat. La recuperación de una entrada debe preservar la publicación y la observación de las entradas sanas, sin reaperturas solapadas. Fuente: [multi_room_audio_loop.py](../../src/pipeline/multi_room_audio_loop.py#L1358).

## Web y usabilidad

Tu casa ofrece una base más cercana al uso cotidiano: a 390 px mantiene el ancho, comunica indisponibilidad y tiene recorridos de rutinas y memoria conectados. El panel técnico conserva elementos de maqueta junto a funcionalidades reales; el aviso global lo reconoce, pero un usuario todavía debe distinguirlos durante la operación.

| Hallazgo | Evidencia | Criterio de mejora |
|---|---|---|
| `ack`, `dismiss`, `restart`, `probe` y comprobaciones sin handler | Botones de Alertas, Sistema y LLM; confirmar `clear cooldown` sólo cambia estado local. | Operación autorizada con progreso y resultado, o control deshabilitado con motivo. |
| Indicadores de ejemplo y datos viejos | Uptime/build literales; ante fallo se conserva mock; varios inventarios sólo se hidratan al iniciar sesión. | Fuente, fecha y caducidad por componente; fallos muestran indisponibilidad o última observación real vencida. |
| Navegación lateral sin teclado | Nueve `DIV` con `tabIndex=-1`, sin rol ni manejo de teclado. | Enlaces o botones semánticos, foco visible y activación por teclado. |
| Panel técnico comprimido en móvil | Layout de 725 px en dispositivo solicitado de 390 px; barra lateral fija de 220 px. | Menú adaptable y contenido/control útil dentro del ancho. |
| Memoria pierde el foco cada segundo | En Chromium, el botón enfocado se reemplazó a los 997 ms; el foco quedó en `BODY` aunque la acción seguía visible. | Mantener el control o restaurar foco y selección sobre una clave estable. |
| Navegación fragmentada | «Panel técnico» enlaza al dashboard legado; vistas sin URL recuperable ni historial. | Entrada coherente a diagnóstico y Voces; recarga y Atrás preservan la vista. |
| Acceso poco claro | El hogar pide un token administrativo; el login técnico confunde errores HTTP con credencial inválida. | Explicar cómo obtener acceso y distinguir credencial, permiso, configuración y red; definir acceso adecuado a cada uso. |
| Revisión de muestras costosa | Fecha, duración y grupo en la lista; habitación y texto se conocen al abrir cada preview. | Filtros por habitación, fecha y revisión con procedencia y retención explícitas. |

La pérdida de foco de Memoria está reproducida. El problema equivalente señalado en la restauración de campos de Rutinas sigue siendo un hallazgo estático. Los recorridos funcionales pasando no cubren por sí solos estas condiciones de usabilidad.

Fuentes: [shell técnico](../../src/dashboard/frontend/obs/src/app.jsx), [Sistema](../../src/dashboard/frontend/obs/src/views/system.jsx#L66), [LLM](../../src/dashboard/frontend/obs/src/views/llm.jsx#L89), [Memoria](../../src/dashboard/frontend/home/memory.js#L68), [Rutinas](../../src/dashboard/frontend/home/routines.js#L331) y [autenticación](../../src/dashboard/auth.py#L4).

## Audios y evaluación de resultados

El informe del 29 de octubre midió 6.254 archivos históricos y 1.915 previews del dashboard. Esas mediciones no confirman la causa de metalización ni sustituyen muestras posteriores al ajuste de Hall. El token de HA abre HA; la medición nueva de previews de KZA sigue pendiente de su credencial propia.

Hay dos límites que cambian cómo evaluar el reconocimiento:

- **Los intervalos difieren:** el preview conserva hasta seis segundos, mientras el texto puede describir hasta treinta segundos. Calcular WER contra ese texto completo inventaría deleciones. Referencia e hipótesis deben abarcar el mismo audio.
- **El canal no figura en la revisión:** la señal de reconocimiento ASR y la de conversación tienen procesamiento distinto. `PP_MIN_NN` afecta conversación; no se puede inferir una mejora del STT por cambiar ese parámetro sin identificar la procedencia y comparar condiciones iguales.

El analizador existente admite una primera lectura acotada de hasta 512 previews recientes, con métricas por habitación y tiempos HTTP, descartando WAV y texto antes de confirmar sus recibos. Esa muestra cronológica sirve para diagnóstico; no es un corpus representativo de voces o condiciones. Los datos ordinarios conservan su vencimiento original, con un máximo de 48 horas desde la captura; pueden vencer antes o retirarse. Una copia no renueva ese vencimiento.

Para puntuar la voz hay que separar cuatro resultados:

| Resultado | Medición necesaria |
|---|---|
| Señal | Nivel, indicios de saturación, silencios internos, saltos y espectro con canal y DSP identificados. Estas métricas no equivalen a SNR acústico ni a naturalidad. |
| Reconocimiento | Referencia humana del mismo intervalo, WER/CER, inserciones, deleciones, sustituciones, vacíos sobre habla y alucinaciones sobre silencio. |
| Tarea | Intención y entidad correctas, operación permitida, efecto final observado y respuesta que coincide con ese resultado. |
| Experiencia | Fin de habla, endpoint, STT, decisión, efecto HA, primer sonido y fin/cancelación; p50/p95 por tarea y condición. Claridad y metalización requieren comparación perceptual. |

El WER existente es macro por frase y reponderado por buckets; debe rotularse así y acompañarse de denominadores y, cuando corresponda, WER micro por palabras. Otro STT puede ayudar a revisar discrepancias, pero no reemplaza una referencia validada.

La agrupación de voces ya tiene un [protocolo](../plans/2026-10-27-h3-voice-grouping-evaluation.md): al menos tres voces consentidas, sesiones de calibración y evaluación separadas, condiciones adversas y criterios previos de uniones, precisión, recuperación y cobertura. Nombrar un grupo no demuestra identidad.

Fuentes: [preview de voz](../../src/users/voice_capture.py#L37), [payload de revisión](../../src/users/voice_ledger.py#L229), [analizador actual](../../tools/analyze_dashboard_audio.py#L63), [WER](../../tools/ambient_wer.py#L55) y [ensayo de Hall](2026-10-29-hall-noise-trial.json).

## Validación de software

| Comprobación | Resultado | Evidencia privada |
|---|---|---|
| Python completo | 5.765 pasadas, 1 `xfail`, 0 fallos y 0 errores; 500,48 s. | `data/project-assessment-20261102/full-tests.log` y `full-tests.xml` |
| Dependencias fijadas | 123 paquetes compatibles, Python 3.13.12 y Torch CPU. | `dependency-check.log` y logs de instalación del mismo directorio |
| Dashboard HTTP y WS | Smoke pasando. | `dashboard-smoke.log` |
| Chromium adicional | Controles, memoria, privacidad de live y pausa ambiental pasando, con fuentes sintéticas locales. | `additional-browser-results.json` y cuatro logs de navegador |
| Reproducciones de auditoría | Dos defectos de recordatorios y pérdida de foco de Memoria observados. | `core-probes.json` y `memory-focus-reproduction.json` |
| Frontend y lint reutilizados | 80 pruebas Node y Ruff pasando; código de aplicación sin cambios desde esa comprobación. | `data/project-assessment-20261101/frontend-tests.log` y `ruff.log` |
| Voces en Chromium reutilizado | Reproducción, carga, grupos y retirada al navegar o perder sesión pasando con fuentes sintéticas. | `data/project-assessment-20261101/browser-voice-tests.log` |

La suite completa corrió con hubs de modelos offline, baja prioridad de CPU y los límites de cada prueba sin ampliar. El único `xfail` es una limitación documentada del regex ante omisión del verbo en una orden de dos habitaciones. Hubo 34.080 advertencias, incluidas deprecaciones de fechas UTC, Torch y WebSockets; conviene reducir ese ruido para conservar señales útiles.

Los dos fallos históricos de la suite no se reprodujeron en esta corrida. Eso aporta una ejecución integrada verde, pero no demuestra sus causas ni sustituye una prueba física de voz, reconocimiento o actuación. Tampoco convierte los ejemplos sintéticos de la web en observaciones reales.

## Orden de mejora y aceptación

| Orden | Trabajo | Cómo se acepta |
|---:|---|---|
| 1 | Estabilizar fuentes y diagnóstico | Cocina entrega audio útil; una recuperación lenta no detiene heartbeat; WS recuperado o degradación explícita; web con observaciones reales fechadas. |
| 2 | Completar recordatorios, timers y rutinas | Entrada por el grafo real, persistencia común con web, cancelación del objeto correcto y entrega confirmada o pendiente/perdida explícita. El reinicio conserva las tareas. |
| 3 | Corregir tareas y navegación de la web | Acciones conectadas o deshabilitadas, foco estable, teclado completo, panel útil a 390 px y errores de acceso comprensibles. |
| 4 | Medir audio y éxito de tareas | Muestras y referencias alineadas, canal/DSP conocido, resultados por condición y comparación perceptual del ensayo de Hall. |
| 5 | Conectar alertas útiles | Fuente HA seleccionada y evento simulado que llega a la salida prevista con recibo, antes de aceptación física. |
| 6 | Cerrar candidato y documentación | Matriz de capacidades vigente, corpus de campo, checks del candidato, artefacto y rollback compatibles. Música queda en el roadmap posterior. |

Cada mejora debe aumentar tareas verificadas y reducir fallos observados. Añadir módulos, contar entidades o producir texto de confirmación no basta para considerar una capacidad terminada.

## Evidencia y límites de acceso

El [JSON de esta evaluación](2026-11-02-comprehensive-assessment.json) conserva 28 hallazgos con fuentes y criterios de aceptación, la matriz de capacidades, puntuación y agregados de HA. La [línea base inicial](2026-11-01-project-assessment.json) se conserva para comparar.

Snapshots, inventario privado, logs completos y capturas locales quedan bajo `data/project-assessment-20261102/` o en la evidencia previa indicada, fuera de Git. El informe versionado contiene agregados, no credenciales, identificadores domésticos, WAV ni transcripciones. La evaluación no ejecutó controles reales ni cambió configuración, servicios, firmware o release.
