# Diseño — KZA adopta el modelo nuevo de luces (grupos `light.grupo_*`) + escenas indexadas

- **Fecha:** 2026-05-31 (revisado el 2026-05-31 tras verificar el estado real de HA y del código en `origin/main`)
- **Proyecto:** KZA (pipeline de voz) — `~/Documents/kza/`
- **Rama de trabajo:** `feat/escritorio-grupos-escenas` (worktree sobre `origin/main` = server `224912d`)
- **Estado:** diseño **aprobado** (Q1=8 cuartos, Q2=arreglar config fantasma, escenas=Approach B). Implementación TDD + deploy al server.

> **Revisión 2026-05-31:** la versión anterior de este spec asumía dos cosas que la verificación
> read-only desmintió. Se corrigen acá con evidencia (ver §"Correcciones al spec original").

## Estado real de HA (verificado read-only, server `localhost:8123`, 353 entidades)

- **Los 8 grupos `light.grupo_*` YA existen y están enabled**, friendly names limpios:
  `grupo_living`(9), `grupo_cocina`(5), `grupo_cuarto`(2), `grupo_bano`(2), `grupo_balcon`(2),
  `grupo_escalera`(5), `grupo_escritorio`(4, fn "Escritorio"), `grupo_pasillo`(3).
- **Conviven** los grupos Hue room/zone (`light.living`, `light.cocina`, `light.cuarto`,
  `light.bano`, `light.balcon`, `light.escalera`, `light.pasillo`, `light.hogar`) — enabled. KZA
  hoy depende de ellos.
- **Conviven** los del Z2M del escritorio: `light.escritorio_2` (fn "Escritorio 4", 3 bombillas) y
  sus miembros `light.escritorio1/2/3`. El `light.escritorio` viejo (huérfano Hue) **ya no aparece**
  en `/api/states`.
- **5 escenas globales "modo"** existen: `scene.cine`, `scene.lectura`, `scene.calida`,
  `scene.fria`, `scene.relax`. (Además hay decenas de escenas por-cuarto `scene.<room>_<x>` y
  `scene.hogar_*` — fuera de alcance.)
- **El escritorio físicamente** solo tiene: luces Z2M, `binary_sensor.escritorio_motion`,
  cámara Blink (`camera.escritorio` + `sensor.blink_escritorio_temperature` +
  `sensor.blink_escritorio_wi_fi_signal_strength`) y `switch.escritorio_camera_motion_detection`.
  **No hay** `climate.escritorio_ac`, ni `media_player.escritorio_*`, ni `sensor.temperature_escritorio`.

## Estado real del código (verificado en `origin/main` `224912d` = server)

- El sync es **`scripts/sync_ha_to_chroma.py`**: `is_group_entity(eid) = eid in KNOWN_GROUPS`;
  por default saltea todo lo que no sea grupo. `KNOWN_GROUPS` incluye `light.escritorio_2`. El
  `VLLMClient` ya lee la API key del entorno (`VLLM_API_KEY`/`LLAMA_API_KEY`/`OPENAI_API_KEY`).
- **El path de rutinas por voz NO está cableado:** `VoiceRoutineHandler` (que tiene los patrones
  `"modo "`/`"activá "` para `intent="execute"`) **no se instancia en ningún lado** del repo;
  `main.py:345` solo construye `RoutineManager` (legacy: create/delete/list, **sin execute**) y
  `RoutineScheduler` entra como `None`. ⇒ hoy "modo cine" no ejecutaría nada por rutinas.
- **El fast-path vectorial sí funciona:** `request_router.py:975` ejecuta el comando resuelto vía
  `self.ha.call_service_ws(command["domain"], command["service"], …)`. El grammar ya tiene el
  intent `scene_activate` (`src/nlu/regex/vocab.py`) y `ha_client` ya maneja el dominio `scene`.

## Objetivo

1. **Cuartos:** KZA controla los 8 cuartos vía `light.grupo_*` (consistente con Alexa, robusto a la
   migración Hue→z2m). Habilita deshabilitar los grupos Hue después (un solo modelo).
2. **Escenas:** "modo cine / lectura / cálida / fría / relax" disparan las 5 escenas HA por el
   **fast-path vectorial** (Approach B — ver §"Correcciones").
3. **Config real del escritorio:** apuntar a entidades que existen; comentar las inexistentes.

## Diseño

### Parte 1 — Grupos: detectar por prefijo `light.grupo_` + re-index (8 cuartos)

**Archivo:** `scripts/sync_ha_to_chroma.py`

- `is_group_entity(entity_id, friendly_name)` → `return entity_id.startswith("light.grupo_") or entity_id == "light.hogar"`.
  Borrar `KNOWN_GROUPS` (YAGNI). Los grupos Hue room/zone dejan de tratarse como grupos.
  **Excepción `light.hogar`** (whole-home, 29 bombillas): se preserva porque NO existe
  `light.grupo_hogar` y es el target de "prendé/apagá toda la casa". Si HA lo deshabilita,
  desaparece de `/api/states` (no rompe). Caught en verificación adversarial.
- **No** tocar `decode_individual` / `GROUP_PREFIX_MAP` (bombita individual `l1`… sigue con
  `--include-individual`). **Preservar** el fix de auth del `VLLMClient`.
- Los friendly_name de los grupo_* ya son limpios (`Living`, `Cocina`, …); `resolve_areas()`
  resuelve por `area_name(entity_id)`. Frases generadas correctas.
- **Re-index:** `python scripts/sync_ha_to_chroma.py --wipe` (borra solo `home_assistant_commands`,
  NO `home_assistant_routines`). Depende de un endpoint LLM (`:8101` con key sourceada) + BGE-M3 +
  token HA.

### Parte 2 — Escenas: indexar las 5 escenas globales como comandos (Approach B)

**Archivo:** `scripts/sync_ha_to_chroma.py` (función dedicada, integrada en `main()`).

- Allowlist de 5 escenas globales: `scene.cine`, `scene.lectura`, `scene.calida`, `scene.fria`,
  `scene.relax`. (Las por-cuarto y `scene.hogar_*` quedan fuera.)
- Función `build_scene_specs()` que emite, por cada escena, un command spec con:
  `domain="scene"`, `service="turn_on"`, `entity_id="scene.<x>"`, `intent="scene_activate"`, y
  **frases curadas con encuadre "modo/escena"** para evitar colisión con el color_temp por-cuarto:
  - cine: "modo cine", "poné modo cine", "activá la escena cine", "ponela en cine".
  - lectura: "modo lectura", "escena lectura", "activá lectura", "luz de lectura".
  - calida: "modo cálido", "escena cálida", "poné todo cálido", "luz cálida en toda la casa".
  - fria: "modo fresco", "escena fría", "poné todo frío", "luz fría en toda la casa".
  - relax: "modo relax", "escena relax", "activá relax", "ponela en relax".
- Se indexan en `home_assistant_commands` junto a las luces (un solo `--wipe` cubre todo).
- **Anti-colisión (test obligatorio):** "poné **la cocina** fría" debe seguir resolviendo a
  `light.grupo_cocina` set_color_temp (lleva room explícito), NO a `scene.fria`.

### Parte 3 — Config real del escritorio + consistencia de `default_light` (`config/settings.yaml`)

Escritorio (Q2 = "arreglar y comentar"):
- `default_light: light.escritorio_2` → `light.grupo_escritorio`.
- `motion_sensor: binary_sensor.motion_escritorio` → `binary_sensor.escritorio_motion`.
- `temperature_sensor: sensor.temperature_escritorio` → `sensor.blink_escritorio_temperature`.
- `default_climate: climate.escritorio_ac` → **comentar** + `# TODO: no existe físicamente`.
- `default_media_player` + `tts_speaker` (`media_player.escritorio_*`) → **comentar** + TODO.

Consistencia de los otros rooms con mic (sobreviven al disable de los Hue; los `grupo_*` ya existen):
- living `light.living` → `light.grupo_living`; cocina `light.cocina` → `light.grupo_cocina`;
  bano `light.bano` → `light.grupo_bano`.
- **hall:** `light.hall` **también es fantasma** (no existe en HA) → `light.grupo_pasillo`
  (hall≈pasillo, alias ya lo incluye). **Flaggeado** por si el usuario prefiere otra cosa.

## No-objetivos (YAGNI)

- No tocar asignación de GPUs (cuda:0-3) ni la cadena de DI de `src/main.py`.
- No cablear `VoiceRoutineHandler`/`RoutineScheduler` (Approach B no lo necesita).
- No crear archivos de config nuevos (todo en `config/settings.yaml`).
- No cambiar el feature de bombita individual (`l1`, `cu2`, …).
- No indexar las escenas por-cuarto ni `scene.hogar_*` (solo las 5 "modo" globales).
- No deshabilitar los grupos Hue en este cambio (es un paso lado-HA posterior, tras validar).

## Tests (TDD, pytest, mocks en `tests/mocks/`)

- **Unit `is_group_entity`:** `light.grupo_cocina`→True; `light.cocina`→False; `light.escritorio_2`→False;
  `light.l1`→False.
- **Unit selección del sync:** dado `/api/states` mock con los 8 grupo_* + bombillas + grupos Hue +
  escritorio_2, `selected` (default, sin `--include-individual`) = exactamente los 8 `light.grupo_*`.
- **Unit `build_scene_specs`:** 5 escenas → cada spec con `domain="scene"`, `service="turn_on"`,
  `entity_id="scene.<x>"`, `intent="scene_activate"`, ≥3 frases con encuadre modo/escena.
- **Unit anti-colisión:** las frases de escena NO contienen room explícito; "la cocina fría" no es
  una frase de `scene.fria`.
- **Smoke `--dry-run`:** lista los 8 grupos + 5 escenas y genera frases sin persistir.
- **Voz (manual, post-deploy):** "prendé el escritorio", "poné la cocina fría" (→ grupo_cocina),
  "modo cine" (→ scene.cine), "modo lectura", "modo relax".

## Verificación de aceptación

1. Tras `--wipe`: "cocina"→`light.grupo_cocina`; 0 docs apuntan a grupos Hue ni a `light.escritorio_2`.
2. "escritorio"→`light.grupo_escritorio` (incl. on/off/dim).
3. "modo cine/lectura/relax", "escena cálida", "escena fría" → `scene.turn_on scene.<x>`.
4. "poné la cocina fría" → `light.grupo_cocina` set_color_temp (sin regresión por las escenas).
5. `settings.yaml` del escritorio: 0 entidades fantasma activas (climate/media comentados con TODO).
6. "prendé/apagá toda la casa" → `light.hogar` (whole-home preservado).
7. (Lado HA, posterior) los grupos Hue room/zone quedan deshabilitables sin romper KZA.

## Riesgos

- Re-index cuesta GPU/minutos y depende del endpoint LLM (`:8101` con key) + VRAM cuda:1 apretada
  (~0.8-1.5GB libres) → **preflight VRAM** + parar `kza-voice` antes (libera lock de chroma).
- Colisión escenas↔color_temp por-cuarto: mitigada con frases "modo/escena" + test de anti-colisión.
- `hall` → `light.grupo_pasillo` es una inferencia (hall≈pasillo); reversible, flaggeada.
- Deshabilitar los grupos Hue es reversible (registry); hacerlo solo tras validar KZA por voz.

## Verificación adversarial (workflow 19 agentes) — resultados

- **Config validada contra HA real:** las 8 referencias nuevas existen; los 7 fantasmas
  comentados confirmados inexistentes.
- **Colisión escenas↔luces:** test empírico con BGE-M3 → "poné la cocina cálido" resuelve a
  `light.grupo_cocina` (sim 0.945) por encima de `scene.calida` (0.775). ✓ light gana.
  Mitigado además acotando las frases de calida/fria a "escena/modo X" (sin "todo"/"casa").
- **Regresión whole-home (`light.hogar`):** caught en self-review → preservado.
- **Falsos positivos descartados por trazado:** `src/nlu/regex/vocab.py` (LIGHT_ENTITIES →
  `light.escritorio`/etc) y `src/rooms/room_context.create_default_rooms()` apuntan a
  entidades pre-grupo, PERO **NO son regresiones de runtime**: el `RegexExtractor` que consume
  vocab **no tiene caller vivo** en el pipeline (solo tests + `__init__`), y `create_default_rooms`
  solo se usa en el CLI `--detect`. El entity ejecutado sale SIEMPRE del vector search
  (`request_router.py:978`), probado por logs reales del server (`[HA-CALL] …@light.escritorio_2`).

## Follow-ups (dead code, NO bloquean — no se tocan en este cambio)

- `src/nlu/regex/vocab.py` LIGHT_ENTITIES y `create_default_rooms()` → actualizar a `light.grupo_*`
  SI se revive el `RegexExtractor` o el CLI `--detect`. Hoy inertes en runtime.
- Slot pollution: "modo cálida" extrae `color_temp_kelvin` que `scene.turn_on` ignora (inocuo).
  Filtrar slots por domain sería un cambio en `request_router`/`slot_extractor` (fuera de scope).
- Test de anti-colisión a nivel vectorial (BGE-M3): impráctico en el venv local (sin chromadb/GPU);
  el test unitario del spec (frases sin room) está cubierto; la validación vectorial se hace por
  voz en el server. Empíricamente ya validado por el workflow.

## Correcciones al spec original (transparencia)

| Spec original asumía | Realidad verificada |
|---|---|
| "Las escenas se disparan con 5 rutinas KZA (`RoutineScheduler.register_routine` + `VoiceRoutineHandler`)" | `VoiceRoutineHandler` **no se instancia** en el repo; `RoutineScheduler=None`; `RoutineManager` no tiene execute. El path de rutinas por voz **no está cableado** → se elige **Approach B (indexar escenas)** por el fast-path vectorial probado, sin tocar `main.py`. |
| "El escritorio quedará en `light.grupo_escritorio`" (implícito) y la config era válida | La config del escritorio apuntaba a **entidades fantasma** (climate/media/sensores inexistentes) y `default_light` estaba en `light.escritorio_2`. Se corrige en Parte 3. |
| KNOWN_GROUPS / `light.escritorio_2` como modelo del escritorio | Convergemos al modelo uniforme `light.grupo_*` (8 cuartos), consistente con Alexa. |

## Punteros

- Notion pág 10 ("Domótica local") — modelo HA nuevo (fuente cross-project).
- HA: `packages/luces_usabilidad.yaml` + `scenes.yaml` + `custom_components/adaptive_lighting/`.
- Memoria: `project_lights_zigbee2mqtt_migration_2026-05-31`.
