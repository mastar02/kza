# Rediseño de NEXO: roadmap y handoff (2026-11-24)

Estado para retomar en una sesión nueva. La fuente de diseño se conserva fuera de Git en
`<repo>/kza-nexo-rediseno/` (`PROMPT.md`, `prototipo/`, `modelo/casa.glb`). El respaldo local
se llama `kza-nexo-rediseno.zip`. `<repo>` es el checkout principal de KZA.

## Dónde se trabaja

- Worktree de integración (`<worktree>`): `<repo>/.claude/worktrees/nexo-rediseno`.
- Rama: `feat/nexo-rediseno`, desde `recovery/production-2026-10-25` (`330aaee4`). Sin push, sin PR.
- El checkout principal queda en `recovery/production-2026-10-25`; no se edita desde ahí.
- Rama sobrante a borrar al cerrar: `worktree-nexo-rediseno` (nació de `origin/main` por error; sin trabajo).
- Ejecutar los comandos desde `<worktree>` y usar rutas absolutas en los asientos de trabajo.

## E1a–E1g implementadas

| Commit o etapa | Contenido |
|---|---|
| `448e0d07` | three r170 addons (GLTFLoader, OrbitControls, BufferGeometryUtils), Space Grotesk, `casa.glb`, `.glb` en `build_runtime.py` |
| `430cc738` | licencia MIT de los addons como `LICENSE.txt` (para que entre en la release) |
| `1a1e60da` | E1a: visor 3D del escaneo + matcher ambiente↔catálogo + parches como datos + test `test_nexo_house_map.cjs` |
| `2969e246` | hooks: ruta del guard independiente del cwd (`$CLAUDE_PROJECT_DIR` / `git rev-parse`) |
| `35019f59` | Merge E1b: tokens visuales, selector de tema y componentes compartidos |
| `28587551` | Merge E1c: navegación única de siete secciones según permisos |
| `f371b511` | Merge E1d: panel de Casa, escenas del catálogo, nuevo encuadre 3D y retiro del eslogan y el pie |
| `03bbd1a5` | Merge E1e: correcciones del pregate para pérdida de WebGL, nombres, luminancia y pines |
| `8bfda862` | Merge E1f: lenguaje de recibos y estado vacío, verificación del empaquetado de Casa |
| `d9a598d8`, `a94a7043`, `84964278` | E1g, correcciones de QA para escenas, ambientes y harness. Merge local a cargo del orquestador. |

Después de E1g el dueño pidió un mapa 3D más grande y fácil de tocar. E1h (`38ce2df2`) agranda el
lienzo, sube la inclinación a unos 57° y baja el corte de paredes a 1,05 m. Cualquier superficie de un
ambiente lo selecciona, con un radio de captura de 44 px, y los carteles quedan compactos. Una prueba
real (`browser_nexo_house_picking.py`) acierta 360 de 360 clics en los nueve ambientes.

## Ola 1 (E2, E3, E5, E6) integrada en local

Base común `d770cd56` (E1 más componentes compartidos `.row-*`, `.pick`, `.field`, `.addrow`, `.time`,
`.empty` y las hojas `rutinas.css`, `tareas.css`, `personas.css`, `memoria.css`). Las cuatro etapas
corrieron en paralelo con un candado de Chromium (`~/.local/state/kza-router/nexo-browser.lock`).

| Commit | Contenido |
|---|---|
| `c5885746` | Merge E2: Rutinas con filas, interruptor, propuestas arriba, vista previa antes/después y campo de condición libre con ayuda |
| `09e2bad7` | Merge E6: Memoria, Mis recuerdos y Mi voz; «Mis recuerdos» es el destino directo del hogar vinculado |
| `550b9d74` | Merge E3: Tareas (estado Hecha/Pendiente como texto) y panel de Avisos, que sigue en Casa |
| `678b5627` | Merge E5: Personas, Voces y accesos por persona (Fase B) |
| `681da3a0` | Corrección de un paréntesis de más en `personas.css:6` (el navegador descartaba la regla en silencio) |
| `735a66ab` | Prueba `browser_nexo_css_parse.py`: el parser de Chromium acepta todas las reglas de las hojas nuevas |
| `df1a636c` | Correcciones del pregate: guarda de homónimos por nombre, sin UUID de recibo desde Personas, ids ARIA locales |
| `de88d4f0` | Migración de selectores de `browser_nexo_memory.py` y `browser_nexo_my_voice.py` (ids retirados del DOM) |

Los dos conflictos de `kza.css` se resolvieron a mano: cada etapa retiró sus reglas y se conservaron
`.memory-content` (lo usa Tareas) y la regla de bordes de `.control-field` sin `#tasks-form`.

### Decisiones tomadas durante la Ola 1 (delegadas por el dueño; revisar en el checkpoint)

- **Avisos compartidos** se queda en Casa; solo cambia su presentación.
- **«Mis recuerdos»** no es un filtro: ningún rol tiene a la vez `memory:read` y `memory:own:read`, así que
  Memoria abre directo en esa vista para el hogar vinculado, sin selector de una opción. El operador ve
  «Recuerdos del hogar».
- **Fase B de Personas** (Voz y Acceso dentro del detalle) solo para el operador y con la intersección de
  permisos, comprobada antes de crear el bloque y de abrir el owner. Hogar y evaluador no ven nada nuevo.
- **Acceso por persona ante homónimos:** si hay otra persona con el mismo nombre normalizado (sin importar
  el nivel) no se ofrece alta ni modificación: se remite a Sistema → Accesos. Se conservan consulta y retiro.
- **Excepciones a «sin IDs visibles»:** (1) «Referencia» ante homónimos en el selector Vincular voz;
  (2) los primeros 8 caracteres en Perfiles en pausa cuando no hay nombre verificado, enlazados con
  `aria-describedby` a un id local por fila. Se retiran cuando el servidor entregue una etiqueta humana
  verificable (alias, rol o fecha de alta).
- **Interruptor global de memoria por voz** sigue en Voces (no se mueve un consentimiento). Personas no
  tiene interruptor de cuenta: no existe una acción reversible.
- **Campo de condición de Rutinas** sigue libre con ayuda de equivalentes (Cerrado = closed...). Un selector
  con «otro valor» queda como propuesta para E8.
- **`data-*` con IDs de registro:** se vaciaron en `people.js`, `access.js` y `voice-consents.js`. Quedan
  conscientemente en `person-voices.js`, `voices.js`, `memory.js`, `my-memory.js`, `routines.js` y
  `tasks.js` (marcadores funcionales preexistentes, no visibles ni parte de un nombre accesible). El gate
  de privacidad evalúa si es aceptable; E8 debe unificar el criterio.

### Estado de los gates de la Ola 1

Pregate: CORREGIR (3 hallazgos, corregidos). QA cruzado: APRUEBA, condicionado a cerrar sobre `de88d4f0`
(13 de 16 mutaciones muertas, una equivalente). UI: CORREGIR por el desborde de 11 px en Rutinas a 320 px
con texto al 200 % y quedó incompleto (el candado se disputó entre gates): repetir. Privacidad: en curso.
En curso: `kza-dev-nexo-w1overflow` (desborde de Rutinas, escenas «sin nombre N» por dispositivo, y la
condición sin traducir en propuestas) y `kza-dev-nexo-w1polish` (huecos de prueba de QA, parser de CSS para
`kza.css` y criterio de homónimos de Vincular voz).

## Ola 1.5 y Ola 2 (ejecución autónoma E2E autorizada por el dueño)

| Commit | Contenido |
|---|---|
| `d0a51f85` | `state.js`: una entrada por línea en `VIEWS` y `VIEW_HASHES` (sin cambio de comportamiento) |
| `7d3b2057` | Hojas `sistema.css` y `actividad.css` enlazadas y cubiertas por las pruebas de empaquetado y de CSS |
| `97462a54` | Ola 1.5: `marco.css`, un único marco de página para todas las vistas migradas (lista única de ids), enlace accesible de Personas |
| `4c6ed17c` | E4 Actividad: tres bloques (acciones de la pestaña, avisos pendientes, última ejecución de cada rutina), permisos por bloque, sin endpoint nuevo |
| `77180b6e` | E7 Sistema: accesos globales y Evaluación con componentes, referencias locales, «Operación N» en vez del UUID del recibo, «Sin lectura confirmada» honesto, getter de sesión limitado |

Verificado en `77180b6e`: node 1032/1032, pytest estructural 28/28, 8 harness más Actividad y la prueba de CSS con exit 0.
Decisiones: Actividad no tiene historial del servidor (solo acciones de esa pestaña) y sus avisos son los
pendientes del owner de Casa; los avisos confirmados quedan como mejora futura. Las anomalías de Tab de
Sistema eran un artefacto de la auditoría con el campo de fecha nativo (los segmentos viven bajo un mismo
elemento activo); el harness ahora comprueba la salida de ese elemento en ambas direcciones.

Cierre (E8) en curso: E8a (retirar `climate.css`, `identity.css`, `style.css`, `kza.css` y el `<style>` inline,
tema por defecto «auto», marco global), E8b (IDs en `data-*`, barrido automático de lenguaje interno,
`ImageBitmap` de Casa, harness tolerantes a la carga) y E8c (auditoría automática de contraste y teclado, tras
E8a). Después gates finales (pregate, UI en asientos pequeños, privacidad, QA cruzado).

## Decisiones del dueño (2026-11-24)

- **Ambientes y dispositivos:** casar con HA y autodescubrir desde el catálogo; no inventar entity IDs.
- **Escenas:** solo las `scene.*` del catálogo. Las tres del prototipo (Noche, Salgo de casa, Apagar toda la casa) quedan fuera.
- **Actividad:** fuente = recibos de acciones, avisos y ejecuciones de rutinas; sin endpoint nuevo. Oculta hasta su etapa.
- **Fuentes e importación:** Space Grotesk vendorizada; los addons de three importan por ruta relativa (sin importmap).
- **Plano 3D local:** el plano real y sus datos quedan en una carpeta ignorada por Git. Solo se sirven como archivos estáticos de `/home` en instalaciones dentro de la red. Verificar a qué interfaz se enlaza el dashboard y respetar el bloqueo de publicación de esta sección.
- **Hook:** corregido, no quitado (el guard bloquea `git add -A`, `/home/kza`, push a main, etc.).
- **Ritmo:** 4 devs en paralelo en la Ola 1; checkpoints con el dueño tras E1 y tras la Ola 1; el resto corre hasta el cierre.
- **Tema:** default oscuro hasta E8 (las vistas sin migrar tienen colores fijos para fondo oscuro); claro disponible en lo ya migrado; claro global en E8.

## Grafo de precedencia

```
E0 ✔ ── E1a ✔ Casa 3D real
              │
   ┌──────────┴───────────┐
 E1b ✔ tokens + tema +   E1c ✔ navegación de 7 secciones +
 componentes (CSS,       registro de secciones (un mount por vista en app.js,
 harness noche)          state.js, index.html)
   └──────────┬───────────┘
        E1d ✔ panel de Casa, escenas del catálogo y retiro del eslogan y pie
              │
        E1e ✔ pregate → E1f ✔ lenguaje y empaquetado → E1g ✔ QA
              │
        CHECKPOINT con el dueño
              │
   ┌─────────┬┴────────┬──────────┐
 E2 Rutinas  E3 Tareas  E5 Personas  E6 Memoria      Ola 1 (en paralelo)
   └────┬────┘            │
    E4 Actividad       E7 Sistema                    Ola 2 (E4 tras E2+E3; E7 tras E5)
        └───────┬────────┘
             E8 Cierre                               tokens 100 %, tema claro global,
                                                     lenguaje interno, suite completa, QA final
```

Propiedad de archivos (evitar choques): E2 `routines.js`; E3 `tasks.js`, `notices.js`; E5 `people.js`,
`voices.js`, `person-voices.js`, `voice-consents.js`, `access.js`; E6 `memory.js`, `my-memory.js`,
`my-voice.js`; E4 módulo nuevo; E7 `evaluation.js` y la parte de Sistema de `access.js`.
Los `*-owner.js` no se tocan (lógica, permisos y confirmaciones se conservan).

## Gates por etapa

`kza-dev` → `kza-pregate` → `kza-ui` (capturas 1440/768/390, día y noche) en paralelo con `kza-privacy`
(E3, E5, E6, E7, E8) → `kza-qa` cross-model. El orquestador integra en `feat/nexo-rediseno`.
Ola 1: tras el merge, rerun de `.cjs`, pytest estructurales y browser tests completos.

## Brief de worker (obligatorio)

SHA exacto de partida, objetivo, archivos permitidos, "nunca `cd`, solo rutas absolutas", ruta del
reporte (`~/.local/state/kza-router/reports/<alias>.md`), checks y criterio de cierre. Ruteo con
`--offline` (no envía el resumen a APIs externas). `hc read <t>` solo muestra la pantalla visible: pedir
siempre el reporte en archivo. Verificar reporte y HEAD antes de aceptar `done`.

## Hallazgos para las etapas siguientes

- Los tests fijan `#rail-nav`, `#section-views`, `[data-view]`, `#logout`, `#help-button` y los ids del visor
  (`house-canvas`, `room-pins`, `view-3d`, `view-top`, `rotate-left/right`, `zoom-in/out`, `reset-view`,
  `expand-map`, `layer-tabs`, `global-scenes`, `map-stage`, `live-badge`). También fijan estructura:
  `test_obs_direct_url.py`, `test_legacy_views_retired.py`, `test_nexo_memory_permission_parity.py`,
  `test_stage_runtime.py`.
- La fixture del test de Casa espera exactamente tres pines enlazados y un ambiente extra en la lista.
- Casa usa nombres genéricos por tipo y ambiente para los dispositivos sin nombre legible y desambigua las escenas en la franja global.
- El visor dibuja las zonas del plano incluso cuando el catálogo carece de un ambiente correspondiente. La selección está disponible en los ambientes enlazados.
- Una corrida adicional de Rutinas agotó la espera del 503 con el cambio de QA aplicado. El reporte local `kza-dev-nexo-e1g.md` conserva el diagnóstico.
- El harness de Voces excluye blobs `image/*` del seguimiento de URLs y puede ocultar una fuga si GLTFLoader falla al cargar una textura (riesgo conocido M3).

## Bloqueo de push

El remoto `mastar02/kza` es público. El dueño aprobó reconstruir la rama desde un árbol limpio.
El orquestador debe verificar el árbol y todos los commits nuevos antes del primer push. El historial
anterior incluye el plano y debe permanecer local.

Guardar `casa.glb` y `house-plan.js` juntos en
`src/dashboard/frontend/home/nexo/private/`. Respaldar ambos en almacenamiento privado del dueño
fuera del repositorio. Para repetir la prueba de picking real, respaldar también
`house-picking-fixture.js` en el mismo lugar. El build empaqueta únicamente el modelo y las
definiciones del plano. Un checkout público conserva la lista de ambientes y los controles de Casa
cuando faltan los archivos del plano.

Para una release que deba mostrar el plano, pasar explícitamente el directorio que contiene
ambos archivos. El build exige que estén presentes, valida la cabecera glTF y registra sus hashes
en `release.json`. El directorio puede ser el del checkout o una copia privada para el build.
Mantener también el artefacto resultante en almacenamiento privado.

```bash
python3.13 scripts/build_runtime.py dist/kza-runtime-cp313.tar.gz \
  --private-plan src/dashboard/frontend/home/nexo/private
```

## Estado final (2026-11-25)

Los pendientes de E8 quedaron cerrados en E8a–E8e, E9a–E9c y E10: tokens al 100 %, tema por defecto
«auto» y claro/oscuro en todas las vistas, un único marco global (`marco.css`), hojas heredadas retiradas,
lenguaje de usuario sin IDs ni jerga, identificadores canónicos fuera del DOM (Map/WeakMap, valores vivos
barridos), liberación de texturas e `ImageBitmap`, pérdida de contexto WebGL, auditoría automática de
contraste y teclado como tests, y harness tolerantes a carga. Actividad (E4) y Sistema (E7) migrados.

Gates finales sobre `f0dec92e..964424a5`: pregate (corregido en E9a/E9b), privacidad APRUEBA (1,8 M
combinaciones rol × permiso sin fallos), UI-A APRUEBA, UI-B reconciliado, QA cruzado APRUEBA (10 de 10
puntos críticos mueren con mutaciones; los hallazgos H1–H5 se cerraron en E10). Verificación final del
orquestador en la cabeza de la rama: 33 archivos `.cjs` con 1038 pruebas, pytest estructural 27/28
(único fallo: `test_compiled_main_and_frontend_outside_checkout`, ver abajo), ocho harness de Casa,
Rutinas, Voces, Accesos, Tareas, Memoria, Mis recuerdos y Mi voz, más Actividad y `css_parse`, todos con
`passed: true`.

Decisiones y excepciones vigentes: Avisos compartidos se queda en Casa; «Mis recuerdos» es el destino
directo de Memoria para el hogar vinculado; Actividad no tiene historial en el servidor (solo acciones de
la pestaña y sesión, más la última ejecución de cada rutina); excepciones a «sin IDs visibles»: «Referencia»
ante homónimos y 8 caracteres en Perfiles en pausa; Voces muestra «Habitación N» cuando el operador no
puede leer el catálogo; el tema por defecto es «auto».

Fallo externo conocido: el merge `26b41dc3` (recovery, +77 archivos de backend de audio/STT/entrenamiento)
entró en la rama sin que el orquestador lo hiciera y rompe
`tests/unit/tools/test_stage_runtime.py::test_compiled_main_and_frontend_outside_checkout`
(«native scope released too early»); falla también en el checkout principal. No toca el frontend de NEXO y
necesita su propio gate. Residuo: `marco.css` usa CSS anidado (Chrome/Edge 112+, Safari 16.5+,
Firefox 117+); decidir un mínimo de navegador o aplanar la hoja.

Pendientes: reconstrucción y verificación del historial antes del push (ver «Bloqueo de push»),
ambiente inicial de Casa, y limpieza de copias de evidencia con `casa.glb` y de ramas locales
`worktree-nexo-rediseno` y `seat/*`.

## Notas históricas de E8

- Harness sensibles a la carga del equipo. Con carga de 30 a 85 en el Mac (observada el 2026-11-24),
  `browser_nexo_home` falló en 2 de 4 corridas incluso en el commit anterior a E1g, por
  vencimiento de las ventanas de frescura del fixture (controles pasan a «Datos desactualizados»,
  quedan deshabilitados y Tab los salta, o falla el chequeo «observations lapsed»). No es una
  regresión del producto. Corregir con relojes o ventanas tolerantes a la carga; mientras tanto,
  verificar con carga baja y repetir antes de concluir. Cerrados en E1: carrera de memoria tras
  `Page.reload` (reintento) y de Rutinas tras reentrar (clic reintentado solo con «Actualizar»
  habilitado); el reloj en milisegundos de Accesos quedó acotado con `Date.now()`.

## Cómo verificar

- Ejecutar `node --test tests/frontend/<archivo>.cjs` por cada uno de los 33 `test_*.cjs` (1038 pruebas).
- Ejecutar pytest con el `.venv` del checkout principal mediante
  `<repo>/.venv/bin/pytest tests/unit/tools/test_stage_runtime.py tests/unit/dashboard/test_legacy_views_retired.py tests/unit/dashboard/test_nexo_memory_permission_parity.py tests/unit/dashboard/test_obs_direct_url.py -q -p no:cacheprovider`.
- Ejecutar `<repo>/.venv/bin/pytest tests/frontend/test_audit_support.py -q -p no:cacheprovider` con Chromium y el candado compartido de navegador.
- Chrome del sistema en headless con SwiftShader. Ejecutar en serie los ocho módulos
  `tests.frontend.browser_nexo_{home,routines,voices,access,tasks,memory,my_memory,my_voice}`
  y verificar `passed: true` en cada resultado. Voces, Accesos y Tareas escriben `result.json`.
  Los demás escriben `browser-result.json`. Para el cierre de E1g, correr Rutinas y Accesos
  tres veces cada uno. Para Casa, usar
  `KZA_NEXO_HOME_EVIDENCE=<dir> <repo>/.venv/bin/python -m tests.frontend.browser_nexo_home`.
- El harness de Casa toma capturas en modo claro y oscuro, y comprueba la luminancia de los pisos
  interiores y su distancia respecto al fondo en ambos temas.
- El escaneo no trae normales ni UV en los pisos: los materiales propios deben ser `flatShading`.
