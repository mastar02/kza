# Roadmap de KZA para aceptación doméstica y ejecución autónoma

El objetivo es completar y verificar KZA para un hogar con varias personas y visitas, usando voz y móvil, hasta aprobar las siete áreas de la evaluación. **10/10 significa cumplir el perfil acordado con evidencia**, incluida la prueba física y de usuarios. No significa perfección en cualquier entorno ni se obtiene compensando un área fallida con otra.

**Estado:** ejecución activada por el objetivo del usuario el 2026-11-02. Se conservan las 16 decisiones, cinco integraciones y metas C01–C21 acordadas. El [manifiesto](2026-11-02-autonomous-roadmap.json) registra el avance; ningún criterio de aceptación se da por medido al iniciar. La [guía de continuidad](../runbooks/2026-11-02-autonomous-execution.md) describe el checkpoint privado y el contrato de evidencia.

La referencia es la [evaluación integral](../research/2026-11-02-comprehensive-assessment.md) de `eb98b70`: 5,1/10 orientativo, 28 hallazgos, 5.765 pruebas pasadas y un `xfail`. Este plan complementa H0–H6; conserva sus restricciones y sus pendientes de privacidad, recursos, mute global y restauración. Una suite verde no cierra esos hitos.

## Decisiones confirmadas

| Tema | Decisión del usuario |
|---|---|
| Personas | Un hogar con varias personas y visitas; voz y móvil. |
| Alcance | Evaluar y priorizar varias integraciones adicionales. |
| Gasto | Desarrollo dentro de la suscripción Codex, sin tope propio mientras esté incluido. Sin cargos nuevos de API, compras de hardware ni ampliaciones de cuota por inferencia. |
| Conversación | Esta versión se centra en tareas domésticas; conversación general postergada. |
| Laboratorio | Simuladores y copias locales durante desarrollo. Pruebas físicas en la ventana final. |
| Voz sin identidad | Timers, listas y recordatorios compartidos, además del control doméstico seleccionado previamente. Información personal sólo con sesión autenticada. |
| Avisos vencidos | Registrar pendiente, mostrarlo y avisar una vez al recuperarse; no ejecutar acciones domésticas atrasadas automáticamente. |
| Rutinas por voz | Una voz sin identidad puede proponer cambios; confirmarlos requiere sesión autenticada y autorizada. Sólo ejecuta rutinas previamente aprobadas dentro de la política y selección domésticas. |
| Participación | Al menos tres personas consentidas; el usuario coordina grabación, referencias, escucha y web. |
| Corpus nuevo | Hasta 30 días desde cada grabación o cierre de su evaluación, lo que ocurra primero; retirada antes si se revoca el permiso. |
| Clientes | iPhone y Chrome; la emulación de laboratorio no sustituye la prueba del teléfono. |
| Entorno de voz | Español rioplatense, wake Nexa, Living/Hall/Escritorio/Cocina, distancias de 1 y 3 metros, ruido habitual y TV. Ambigüedad y solapamiento requieren aclaración o abstención. |
| Respuesta hablada | Conservar la voz actual; priorizar claridad, menos metalización y respuestas breves. |
| Etapa final | Hasta siete días de observación y dos sesiones coordinadas de participantes. Fecha y acciones se revisan con candidato y rollback concretos. |
| Cartera obligatoria | Energía/consumo, compras/tareas HA, calendario doméstico en lectura, multimedia/TV HA y salud/eventos técnicos de cámaras existentes, sin vídeo ni biometría. |
| Criterios de cierre | C01–C21 aceptados; congelar antes de evaluar, sin atribuir resultados todavía no medidos. |

El [registro de preguntas](2026-11-02-autonomous-questions.md) conserva estas respuestas. Los IDs, nombres, datos personales, credenciales y selección concreta de controles permanecen en el entorno privado.

## Alcance doméstico e integraciones elegidas

La base obligatoria comprende control doméstico seleccionado, consultas domésticas y clima, listas, timers, recordatorios, rutinas, alertas existentes, memoria revisable, grupos de voces, privacidad y diagnóstico. Las tareas compartidas tienen actor doméstico explícito; una voz desconocida no se transforma en una cuenta personal. Las operaciones sensibles conservan la política común. Q19 permite proponer cambios de rutinas por voz y ejecutarlas sólo si ya fueron aprobadas; confirmar cambios exige una sesión autenticada y autorizada.

| Integración | Utilidad y evidencia disponible | Alcance propuesto | Dependencia para aceptar |
|---|---|---|---|
| I01 Energía y consumo | Había 33 sensores de energía y 15 de potencia informando estado. | Consultas y resúmenes con unidades, fuente, intervalo y frescura; sin importes sin tarifa ni cortes automáticos. | Sensores revisados e histórico válido cuando se informe consumo por período. Potencia instantánea no equivale a energía. |
| I02 Compras y tareas HA | Una entidad `todo` y servicios `todo`/`shopping_list`. | Lista compartida coherente entre voz, web y HA, con completados y conflictos explícitos. | Lista elegida, autoridad de persistencia e idempotencia. El estado de la entidad no reemplaza la lectura de sus elementos. |
| I03 Calendario doméstico | No había entidad `calendar`; hay alternativa de archivo o fuente privada local/ICS. | Agenda compartida en lectura; calendario personal sólo con sesión y fuente autorizada. | El usuario designa una fuente real. No crear cuentas, escribir calendarios externos ni instalar servicios ajenos automáticamente. |
| I04 Multimedia y TV por HA | Dos reproductores figuraban no disponibles; un `remote` informaba estado. | Estado, reproducción/pausa y volumen según capacidades reales. | Un dispositivo operativo y operaciones/límites elegidos. No se presupone compatibilidad multiroom ni suscripción Spotify. |
| I05 Cámaras existentes | Una cámara informaba estado y tres no estaban disponibles. | Disponibilidad y eventos técnicos de estado/error ya expuestos, sin fotogramas, grabación, movimiento o personas inferidos. | Fuente autorizada y eventos disponibles; no implica instalar Frigate. |

Los contratos de [listas](https://www.home-assistant.io/integrations/todo/), [calendario](https://www.home-assistant.io/integrations/calendar/) y [multimedia](https://www.home-assistant.io/integrations/media_player/) dependen de las capacidades expuestas por cada integración de HA. La [gestión de energía](https://www.home-assistant.io/docs/energy/) distingue lecturas de potencia y energía. La presencia de un dominio no habilita automáticamente todas sus operaciones.

Se evaluaron y se difieren Spotify específico, nuevas instalaciones de Frigate/vídeo persistente, mensajería externa, conversación general y nuevo LoRA: necesitan cuentas, alcance, recursos o costes no confirmados. Codex se usa para desarrollar; no se convierte su sesión en una API de inferencia de KZA. Los adaptadores existentes de proveedores pueden conservarse sin llamadas de pago nuevas en este perfil.

Una integración requerida con dependencia ausente queda **bloqueada**, no aceptada. Cualquier cambio de alcance necesita una decisión registrada; el agente no puede quitarla silenciosamente para obtener la nota.

## Criterios para aceptar cada área

| Área | Punto de partida | Evidencia necesaria para 10/10 en este perfil |
|---|---:|---|
| Web Tu casa | 7 | Recorridos completos, fuente real, permisos, móvil, rendimiento y aceptación en iPhone. |
| Panel técnico | 4 | Datos actuales con procedencia, sin ejemplos operativos ni controles engañosos; diagnóstico y recuperación observables. |
| Usabilidad y accesibilidad | 5 | Teclado, foco, errores, accesibilidad y tareas de usuarios que cumplen el perfil. |
| Núcleo y arquitectura | 7,5 | Fábrica real compartida, recursos acotados, privacidad completa, pruebas y artefacto/rollback reproducibles. |
| Captura y voz | 4 | Cuatro entradas útiles, reconocimiento, wake, entrega, escucha y agrupación aceptados con corpus reservado. |
| Funciones completas | 3,5 | Tareas pedidas que persisten, se cancelan correctamente y tienen efecto/entrega confirmado o incertidumbre explícita. |
| Operación e integraciones | 4 | Fuentes reales, recuperación, estabilidad, selección y aceptación física de cada integración incluida. |

Cada área conserva sus criterios y evidencia individual. Un criterio obligatorio sin medir, un P0/P1 abierto del alcance o una dependencia real ausente impide aceptar esa área. Los porcentajes de tareas implementadas sirven para seguimiento, no para fabricar una nota de madurez.

### Metas nuevas aceptadas en Q15

Estas metas se congelan antes de abrir la reserva. Son exigencias aceptadas, no resultados alcanzados. Se registran cantidades, exclusiones, distribución y condiciones; una muestra pequeña no se presenta como garantía futura. Las precisiones de método siguientes no alteran las cifras aceptadas.

| Criterio | Meta propuesta | Evidencia mínima y límite |
|---|---|---|
| C01 Tarea completa | 100 % de casos obligatorios deterministas desde la fábrica real. | Petición, actor, política, persistencia, efecto/entrega y respuesta; crear/consultar/cancelar/reiniciar para cada familia. |
| C02 Integridad | Cero efectos no autorizados, falsas confirmaciones, reenvíos ciegos o audio antiguo reactivado. | Matriz negativa y de fallos; al menos 200 entradas etiquetadas en el protocolo, con separación de simulación y campo. |
| C05 Candidato | Suite y checks requeridos verdes, sin `xfail` sobre requisitos incluidos. | Commit exacto, dependencias fijadas, arranque validado, artefacto CPython 3.13 y restauración compatible. Entrenamiento nocturno deshabilitado si su utilidad/recursos no están aceptados. |
| C06 Veracidad | Todas las observaciones operativas tienen fuente, fecha y vencimiento; cero mock presentado como real. | Sesión perdida, 403, 409, 503, red, WS, caducidad y efecto incierto. |
| C07–C09 Accesibilidad y uso | WCAG 2.2 AA en el alcance; reflow desde 320 px; al menos 95 % global de tareas prioritarias sin ayuda, sin un participante bloqueado en un recorrido crítico. | Inventario previo de vistas, estados y procesos. Tres personas y diez tareas por persona; todos los intentos preregistrados, resultados por persona/tarea y cero confusiones críticas. Ayuda es indicación externa, no feedback normal de KZA. |
| C10 Carga e interacción | LCP ≤2,5 s, INP ≤200 ms y CLS ≤0,1 en p75 donde sean medibles. En iPhone: contenido útil p75 ≤2,5 s e interacción a pintura p75 ≤200 ms, además de C11 y sin saltos que cambien el objetivo táctil. | 40 recorridos por cliente, frío/caliente separados, dispositivo/versiones y método/precisión preregistrados; instrumentación contrastada con traza o grabación física. Una API ausente no permite inventar CWV ni aceptar el teléfono sin método alternativo fiable. |
| C11 Feedback web | Feedback visible p95 ≤100 ms; preview vigente de hasta seis segundos listo p95 ≤1,5 s en LAN. | 40 interacciones/aperturas por cliente. Inicio en activación y cierre en pintura observada/reproducción posible con gesto nativo; DOM y metadatos no bastan. Separar red, preparación y espera humana registrada. Las aperturas válidas fallidas no se excluyen para mejorar el p95. |
| C12 Reconocimiento | WER micro ≤10 % en claro y ≤20 % en adversos admitidos; cobertura ≥95 %. | Al menos 200 peticiones claras y 100 adversas elegibles, 20 por condición crítica; referencia humana del mismo tramo, macro/CER también publicados. |
| C13 Éxito de tarea | ≥98 % en claro y ≥95 % en adversos admitidos. | Cada familia tiene al menos diez casos; un turno atendido sin realizar la tarea no cuenta como éxito. |
| C14 Wake | Rechazo ≤5 %; falsos wakes ≤0,1 por hora de casa, con desglose por habitación. | 100 wakes intencionales y 48 horas de reloj con las cuatro entradas útiles, no horas-micrófono. Revisar ≥95 % de eventos; los inciertos se conservan y su cota conservadora no puede hacer pasar el umbral. Deduplicar sólo el mismo evento físico corroborado y conservar conteos originales. |
| C15 Demora doméstica | Desde fin de habla, primera respuesta significativa p50 ≤0,8 s y p95 ≤1,5 s; resultado software/observación HA de acciones rápidas p95 ≤2 s. | 100 recorridos simples, con desglose por familia/frío/caliente; un earcon no cuenta como respuesta. Movimientos largos tienen plazo propio antes del test. |
| C16 Entrega | ≥99 % de entregas debidas completas y todos los estados de fallo/parcial/cancelación correctos; interrupción acústica p95 ≤250 ms. | 100 entregas y 20 interrupciones; respuesta larga no cacheada, barge-in y captura real concurrente. Cancelación voluntaria separada. |
| C17 Claridad | Claridad mediana ≥4/5 por participante y cero defectos graves de metalización/recorte en el lote. | 30 respuestas cortas/largas, A/B ciego balanceado y tres personas. Rúbrica interna, no MOS validada. |
| C19 Estabilidad | Disponibilidad observada ≥99,9 % de KZA y de cada entrada requerida. | Soak local y bloque físico fijo de al menos 72 h tras congelación, dentro de siete días. Segundos útiles/segundos del bloque. Inyección preregistrada antes/después; informar también disponibilidad total y fallos planificados/espontáneos. No excluir fallos retrospectivamente ni promediar una habitación ausente. |
| C20 Recuperación | HA/WS ≤60 s desde dependencia sana; micrófono ≤180 s tras reconexión y enumeración. | Mantener heartbeat de las entradas sanas; ensayos locales y guion físico autorizado, incluyendo arranque. |
| C21 Integraciones | Cada contrato incluido funciona con al menos una fuente/dispositivo real elegido. | Datos, permisos, caducidad, fallo y conflictos; un fixture no sustituye la fuente real. |

El objetivo de accesibilidad se basa en [WCAG 2.2](https://www.w3.org/TR/WCAG22/), con revisión manual además de automatización. Los objetivos de LCP, INP y CLS y su p75 se toman de [Web Vitals](https://web.dev/articles/vitals); no implican conformidad ya alcanzada. El objetivo táctil propio para controles principales es 44 px.

La reserva de voz cubre las cuatro habitaciones, ambas distancias y las tres personas. Antes de abrirla se fija la distribución, con al menos diez casos elegibles por habitación, distancia y participante para cada recorrido aplicable, además de los mínimos globales y de condición. Se informan las métricas por estrato: uno fallido o con evidencia insuficiente no queda aceptado por un promedio global. El mínimo H3 de dos habitaciones no sustituye el perfil doméstico de cuatro.

Los intentos de usabilidad se agregan globalmente para el 95 %, con cantidades por persona y recorrido. Un bloqueo crítico se corrige y revalida, conservando el primer resultado. En iPhone se usa el navegador habitual elegido y se registran versiones; no se exige Safari por inferencia ni se sustituye la prueba física por Chromium Linux. Las aperturas de audio separan solicitud, preparación y reproducción, y los fallos se conservan en el resultado.

### Contratos existentes que se conservan

**C03 H2:** heartbeat de 5 ms, retraso p95 ≤25 ms y máximo ≤75 ms; diez repeticiones con productor simulado de 200 ms. Cancelación ≤100 ms, parada simulada ≤200 ms y cleanup cooperativo ≤1 s. PCM exacto y en orden, sin pérdida ni duplicación. Son límites offline y no se sustituyen por percentiles de otro reloj.

**C04 Privacidad:** inventario completo de productores, SDK, copias y participantes; mute global durable y reactivable sin voz; vencimiento/borrado coordinados y restauración sin resurrección. La pausa de AmbientPath no equivale al mute total.

**C18 H3:** al menos tres voces, diez segmentos claros por voz para calibrar y treinta por voz para evaluar; dos habitaciones/distancias, al menos diez ejemplos por condición adversa, orden original y diez permutaciones 0–9. Cero uniones entre personas, precisión de pares ≥0,99, recuperación ≥0,80 y cobertura ≥0,70 en claros elegibles. Los cortos y solapamientos deben abstenerse. Una prueba de voz sin perfil no demuestra generalización a cualquier visitante.

Las políticas de [H2](2026-10-25-h2-complete-voice-responses.md), [H3 y agrupación](2026-10-27-h3-voice-grouping-evaluation.md), cuarentena, separación perfil/identidad y bloqueos de cerraduras/alarmas continúan vigentes. Se preservan el máximo ordinario de 48 h, hechos automáticos de 30 días y recuerdos/perfiles confirmados hasta borrado. El corpus nuevo es una finalidad separada y no renueva ni convierte datos ordinarios en entrenamiento.

## Secuencia de ejecución

| Fase | Trabajo | Salida verificable | Dependencias |
|---|---|---|---|
| P0 Contrato y continuidad | Fijar perfil, cartera, criterios, selección privada y estado reanudable. | Q14/Q15 resueltos, tareas/evidencia trazadas y permisos sin ampliar por inferencia. | Decisiones confirmadas. |
| P1 Grafo y contratos | Fábrica real de main, ciclo de tarea, fuente/frescura, sesiones por rol y procedencia del audio. | APIs, voz y web usan contratos y autoridad comunes. | P0. |
| P2 Recuperación y privacidad | Heartbeat independiente, HA/WS, recursos/SDK/copias, mute global, borrado/rollback y diagnóstico de Cocina. | Fallos simulados aceptados; las verificaciones físicas siguen pendientes de P7. | P1. |
| P3 Funciones domésticas | Timers, creación/cancelación de recordatorios, rutinas persistentes, listas, alertas y consultas locales. | Recorridos del grafo con persistencia, reinicio y entrega, incluidos fallos. | P1; usa garantías de P2. |
| P4 Web y diagnóstico | Operaciones reales, navegación, foco/teclado, móvil, búsqueda/filtros, errores y rendimiento. | Flujos locales completos; métricas de cliente y prueba física se completan en P7. | Contratos P1; concilia P2/P3. |
| P5 Integraciones | Implementar la cartera elegida, permisos y adaptadores sobre fuentes reales o fixtures marcados. | Contratos locales aceptados y fuentes reales designadas; dependencias ausentes visibles. | P1/P3. |
| P6 Candidato | Regresiones, soak local, reservas/evaluador, bytecode, migraciones en copias y rollback. | Candidato y guion concretos revisables; no quedan bloqueos de software requeridos para campo. | P2/P3/P4/P5. |
| P7 Ventana final | Operación por responsable de kza, cuatro entradas físicas, calibración y reserva, iPhone, escucha y observación. | Metas de campo, fuentes, datos compatibles y decisión de promoción o rollback. | P6 y autorización concreta de ventana/acciones. |
| P8 Aceptación | Revisión independiente de las siete áreas, evidencia, guía y estado H0–H6. | Cada criterio incluido aceptado; entrega y cierre del objetivo. | P7. |

El [manifiesto de ejecución](2026-11-02-autonomous-roadmap.json) descompone el trabajo en **50 tareas** y **21 criterios**, vinculados a los 28 hallazgos y a la referencia visual aprobada. P2, P3 y P4 pueden investigarse/revisarse en paralelo tras P1; habrá un solo escritor de código/Git y las suites que miden tiempos se ejecutarán sin carga de pruebas competidora.

Por instrucción del usuario del 2026-11-02, **T047 integra el diseño NEXO en el frontend de KZA dentro del goal activo**. Comienza después de cerrar y subir la unidad actual de administración nativa de T013. La referencia local es `/home/yo/projects/kza-design-reference/nexo-2026-11-02-d686fcd`: `LEEME.txt`, `HANDOFF.json`, código en `site/` y capturas en `previews/`, procedentes del commit `d686fcde52a34c3b786decf167eebdb91dceb6d6`. Se conserva su estética oscura, cian/violeta, tipografía, casa 3D y apartados Hogar, Voces, Personas y Acciones, adaptándolos al backend y al catálogo reales. Los datos y efectos simulados se sustituyen por estados y operaciones autenticadas con resultados verificables.

La vinculación manual perfil vocal–persona requiere revisar e implementar un contrato explícito si falta: confirmar, nombrar o asociar una voz no autentica a una persona, concede roles ni habilita acceso personal. Identidad acústica, cuentas, consentimiento y autorización conservan ámbitos separados. Los ambientes/dispositivos se concilian por identidades del catálogo, sin inferir asociaciones por nombre. T047 coordina los contratos pendientes de T013 y las comprobaciones de T040–T046; su prioridad de inicio no da por terminada P1 ni permite aceptar P4 sin sus dependencias. La integración se desarrolla y prueba localmente; la publicación en producción sigue reservada a una release.

Las dependencias de P1–P6 exigen entregables requeridos **verificados localmente**. Las tareas opcionales no seleccionadas no bloquean ni se marcan implementadas. Los criterios físicos compartidos se aceptan en P7; no se exige una prueba de hardware anterior para poder preparar el candidato. P7 requiere además la autorización concreta de la ventana y P8 exige toda la evidencia de campo.

El plan avanza por dependencias y aceptación, sin prometer una cantidad fija de semanas. Los bloqueos físicos o de fuente se mantienen abiertos. No se reduce el corpus ni se relajan métricas después de ver la reserva para cumplir una fecha.

## Ventana física y corpus

Durante desarrollo se usan simuladores y copias bajo `yo`. No se ensayan efectos en la casa, no se reinician servicios de producción ni se modifica HA u otro proyecto. El agente prepara código, diagnóstico compilado, corpus sintético, evaluador y procedimientos.

Al existir candidato, se presenta un manifiesto con entidades/operaciones exactas, límites, número de ciclos, horario, responsables, riesgos de indisponibilidad, respaldos y restauración. La selección privada previa se conserva y se revisan sus diferencias; no se pide recordar IDs internos. La ventana de hasta siete días sólo se agenda y autoriza sobre ese material.

La secuencia final propuesta es:

1. El operador verifica identidad, canal y habitación de cada micrófono, conecta o diagnostica Cocina y comprueba fuentes/dispositivos de las integraciones. Se instala el candidato dentro del procedimiento de release, con rollback preparado.
2. Primera sesión: calibración y comprobaciones de entrada/salida con participantes consentidos. Se congelan parámetros y revisión del candidato antes de abrir la reserva.
3. Se completa la observación de al menos 72 horas sobre la revisión congelada y se revisan eventos antes de vencer. Los fallos planificados se distinguen de caídas espontáneas y también se informan.
4. Segunda sesión en otro día: reserva independiente, tareas, escucha A/B e iPhone. Se registra efecto HA y comprobación física según la naturaleza de la operación: aplicar una temperatura objetivo no significa que el ambiente la alcance en dos segundos.
5. Se promueve sólo si pasan los criterios, o se restaura el conjunto compatible. Si un arreglo exige cambiar el candidato, se vuelve a validar lo afectado; otra ventana o nueva reserva se acuerdan cuando sean necesarias.

Calibración y reserva usan sesiones completas separadas; copias, recortes y aumentos heredan el conjunto de su origen. No se atribuye a otro STT la autoridad de referencia humana. Los previews de seis segundos no se comparan contra textos de treinta segundos sin alinear.

El corpus nuevo registra finalidad de calibración/evaluación, consentimiento, captura, vencimiento, canal/DSP, condiciones, referencias y revisión en un manifiesto privado. Vence en el menor de captura+30 días, cierre formal de su evaluación o revocación, incluyendo copias/derivados. No autoriza LoRA, clonación o conservación indefinida de embeddings. Las exclusiones están definidas antes de medir y no dependen de que el modelo haya fallado.

## Autonomía y reanudación durante semanas

La ejecución utilizará un objetivo lógico persistente con checkpoints; su continuidad depende de entorno, credenciales y cuota. Goal mode permite objetivos verificables y reanudación, manteniendo los límites de acceso. [Documentación oficial de trabajo prolongado](https://learn.chatgpt.com/docs/long-running-work).

Se verificó localmente Codex CLI 0.166.0 y el soporte de `codex exec resume` por ID. El ID de sesión real se guardará en privado al activar; no se usará una selección ambigua de otra sesión. Este plan no instala un daemon ni activa automatizaciones.

El objetivo de arranque, cuando se solicite expresamente, será ejecutar `docs/plans/2026-11-02-autonomous-roadmap.json` bajo `yo` hasta aceptar las siete áreas con evidencia. Incluye presupuesto de suscripción, ámbito doméstico, privacidad, un escritor, límites GPU y campo sólo en ventana autorizada. El estado y los commits se guardan por unidad; se avanza con tareas independientes mientras una rama espera datos o intervención. No se completa el objetivo mientras falten criterios humanos o físicos. El mismo texto está disponible como `activation_prompt` en el manifiesto.

### Ciclo de trabajo

1. Confirmar usuario, checkout, rama, cambios ajenos y reglas vigentes. Leer el estado de la tarea y su última evidencia con la recuperación enfocada acordada.
2. Verificar si un comando o cambio de la unidad anterior ya terminó antes de repetirlo. No duplicar trabajo por una interrupción de la conversación.
3. Elegir una tarea con dependencias satisfechas. Implementar una unidad coherente y verificar su comportamiento con pruebas pertinentes; ampliar comprobación cuando haya cambios transversales o nueva incertidumbre.
4. Registrar commit, comando, resultado, entorno y ruta privada de evidencia. Actualizar estado, guardar y subir la rama. Una tarea puede estar implementada sin quedar aceptada físicamente.
5. Revisar de forma independiente los cambios críticos, elegir la próxima tarea y continuar sin preguntar «¿seguimos?» dentro del perfil acordado.

El manifiesto distingue `planned`, `in_progress`, `implemented`, `verified_local`, `waiting_external`, `accepted` y `deferred_by_user`. Cada estado conserva evidencia; marcar una tarea `accepted` exige sus criterios, incluidos los humanos que correspondan. Las decisiones de gasto, datos, permisos, GPU, alcance y producción se consultan con una propuesta concreta. Los problemas técnicos ordinarios se resuelven dentro de la autonomía acordada.

### Límites y bloqueos

| Situación | Respuesta de la ejecución |
|---|---|
| Se termina cuota o conectividad | Conservar checkpoint y usar reanudación soportada; no cambiar identidad, comprar créditos ni prometer continuidad automática. |
| Falta una credencial o autorización | Bloquear esa rama y continuar trabajo independiente; no inspeccionar secretos de otra cuenta ni reutilizar una credencial para otro servicio. |
| Repite el mismo fallo sin información nueva | Reintentos acotados con backoff; tras tres intentos, revisar causa y estrategia. No insistir hasta provocar otro bloqueo de HA. |
| Falta hardware útil o fuente real | Mantener el criterio de campo pendiente; entregar diagnóstico y preparación para el responsable. No comprar ni declarar aprobado el simulador. |
| Aparece una decisión nueva de producto o recursos | Presentar opciones e impacto, sin interpretar silencio como consentimiento ni rebajar metas. |
| Se detecta trabajo ajeno o cambio de rama | Protegerlo y reconciliar con el responsable; ningún agente cambia el checkout por debajo de otro. |
| Se cumple todo el perfil | Publicar evidencia y entrega, y cerrar el objetivo. No continuar indefinidamente sólo para consumir suscripción. |

Se prevé actualización al completar cada unidad y un resumen diario en este hilo mientras haya trabajo activo. Los bloqueos se comunican al detectarse. No se habilitan mensajes externos, túneles, índices RAG o servicios adicionales como efecto secundario.

## Restricciones conservadas y cierre del plan

El desarrollo, pruebas, construcción y Git siguen bajo `yo` en este checkout. Producción sigue bajo `kza` y sólo cambia mediante release del artefacto compilado y recursos necesarios. No se reasignan GPUs, no se desarrolla en `/home/kza/kza` o `/home/kza/app`, no se borran fuentes necesarias para rollback ni se opera infraestructura de otros proyectos.

La configuración de aplicación continúa centralizada en `config/settings.yaml`; el manifiesto de este plan es seguimiento de trabajo. Los datos privados, WAV, referencias, identidades y credenciales permanecen fuera de Git/RAG. Las decisiones de retención existentes se mantienen salvo la finalidad nueva y acotada del corpus confirmada en esta conversación.

Q14, Q15 y Q19 quedaron confirmadas: cinco integraciones, metas de aceptación y autoridad de rutinas por voz. El objetivo del usuario activó la ejecución el 2026-11-02. Q16–Q18 se resuelven con inventario, candidato o acceso concretos cuando hagan falta y no bloquean la preparación independiente. La creación original del roadmap fue planificación; la activación y la evidencia posterior constan por separado en el manifiesto.
