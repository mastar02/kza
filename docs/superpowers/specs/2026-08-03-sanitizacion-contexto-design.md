# Sanitización del contexto por sesión — Diseño

**Fecha:** 2026-08-03
**Estado:** aprobado, sin implementar
**Objetivo:** bajar el arranque en frío de KZA de 90.130 a ~40.000 tokens sin perder ninguna capacidad que se use.

## Problema

Cada sesión de KZA arranca con 90.130 tokens de contexto antes de que el usuario escriba nada. La
medición sale del primer request de la sesión `4e2c30cb` (`cache_creation_input_tokens` del `.jsonl`
del transcript), no de una estimación.

Desglose medido:

| Fuente | Tools | Tokens | % |
|---|---:|---:|---:|
| MCP notion | 24 | 20.485 | 23% |
| MCP trello | 57 | 8.894 | 10% |
| MCP github | 26 | 4.242 | 5% |
| MCP claude-in-chrome | ~20 | ~4.000 | 4% |
| MCP memory | 9 | 2.875 | 3% |
| MCP context7 / prompt-refiner / homelab-search / konsi-code-search | 8 | ~1.400 | 2% |
| Listado de 63 skills globales | — | 6.712 | 7% |
| Skills de plugins + tipos de agente | — | ~3.300 | 4% |
| MEMORY.md | — | ~4.960 | 6% |
| CLAUDE.md | — | ~2.740 | 3% |
| Base de Claude Code + tools built-in | — | ~30.500 | 34% |

Los tamaños de MCP se midieron hablando `tools/list` por stdio contra cada server y contando el JSON
devuelto (`scratchpad/mcp_size.py`), no estimando.

### Causa raíz

Todos los MCP están declarados en `~/.claude.json` → `mcpServers`, es decir **globalmente**, y cada
uno se usa en **un solo proyecto**. KZA paga las 57 tools de Trello y las 24 de Notion en cada una de
sus sesiones para no llamarlas nunca.

Uso real, medido sobre 1315 transcripts en 29 proyectos (invocaciones `type: tool_use`, descartando
menciones en listas de permisos que se re-inyectan por sesión):

| Server | Llamadas | Proyectos donde se usa | En KZA |
|---|---:|---|---:|
| trello | 585 | konsi | **0** |
| playwright | 133 | agrotrace (123), konsi (10) | 0 (ya deshabilitado) |
| github | 74 | konsi (71), Documents (3) | **0** |
| claude-in-chrome | 48 | homelab-infra (42), konsi (5), agrotrace (1) | **0** |
| notion | 16 | homelab-infra | **0** |
| konsi-code-search | 11 | konsi | **0** |
| homelab-search | 9 | 5 proyectos | 0 |
| prompt-refiner | 2 | prompt-refiner, homelab-infra | 0 |
| **memory** | **0** | **ninguno** | **0** |

En KZA: **0 invocaciones MCP reales en 123 transcripts, del 2026-07-02 al 2026-08-03.**

El mismo patrón en skills: de las 63 en `~/.claude/skills/`, solo dos se invocaron alguna vez en toda
la historia de los 29 proyectos — `codebase-exploration` (4) y `senior-prompt-engineer` (1). Las que
sí se usan (`superpowers:*` con 29+23+22 invocaciones, `pr-review-toolkit:*`, y las built-in) vienen
de plugins y de la base, no de ese directorio.

Nota sobre notion: sus 20.485 tokens no son inherentes a Notion. El server repite el bloque `$defs`
completo (`blockObjectRequest`, `richTextRequest`, `sortObject`, …) en las 24 tools — el mismo texto
24 veces.

## Diseño

### 1. MCP: de global a por-proyecto

`~/.claude.json` → `mcpServers` queda con solo lo de uso disperso. Cada server se declara en el
proyecto donde el dato dice que se usa.

| Server | Destino | Justificación |
|---|---|---|
| trello, github, konsi-code-search | konsi | 585 / 71 / 11 llamadas, todas ahí |
| notion, claude-in-chrome | homelab-infra | 16 / 42 llamadas, todas ahí |
| playwright | agrotrace + konsi | 123 / 10 |
| homelab-search, prompt-refiner | siguen globales | 173 y 338 tok; uso disperso en 5 proyectos |
| memory | **se elimina** | 0 llamadas en 29 proyectos |

KZA queda con context7 (plugin), homelab-search y prompt-refiner. **−40.700 tok.**

`~/.claude.json` también guarda estado de sesiones, costos y flags por proyecto. Se edita con un
script quirúrgico sobre la clave `mcpServers`, con backup previo del archivo completo. No se
reescribe entero.

### 2. Skills: archivar las muertas

Las 61 skills sin uso registrado se mueven de `~/.claude/skills/` a `~/.claude/skills.archive/`.
Se quedan `codebase-exploration` y `senior-prompt-engineer`. Las de plugins no se tocan.
**−6.400 tok.** Reversible con un `mv`.

Existe además `skillOverrides: {"<nombre>": "off"}` como clave de settings a nivel proyecto
(confirmado en el binario 2.1.220, leída desde `localSettings`). No se usa acá: mover el directorio
es más simple y aplica a los 29 proyectos de una vez, que es donde está el ahorro.

### 3. CLAUDE.md — sección "Source of truth cross-project"

`docs/SERVER_CONVENTIONS.md` pasa de "consultar primero" a fuente operativa. Notion se documenta como
consulta bajo demanda, con la línea concreta para reactivarla. Sin este cambio, CLAUDE.md instruye
usar `mcp__notion__*`, una herramienta que ya no estará cargada.

### 4. MEMORY.md

MEMORY.md está especificado como índice —una línea por memoria, `- [Título](archivo.md) — gancho`— y
creció a párrafos de 3-5 líneas con las lecciones inline. El detalle ya vive en los archivos
individuales, así que comprimirlo a índice no pierde información: queda a un `Read` de distancia.

19.838 → ~8.000 chars, **−3.000 tok**.

Criterio de poda: las entradas marcadas ✅ (cerradas/mergeadas) colapsan a una línea con su pointer.
Las abiertas (⚠️, 🛒, sin marca) conservan el gancho que las hace accionables. Ningún archivo de
memoria se borra.

### 5. Los `.md` del repo

Los 294 `.md` de `docs/` **no cuestan tokens por sesión**: no se cargan en contexto, solo pesan
cuando se leen. Limpiarlos no baja el arranque en frío y queda fuera del objetivo de este spec.

Lo único con costo real: el worktree `.claude/worktrees/kza-dashboard` duplica `docs/plans/` y
`docs/superpowers/plans/`, así que cada Grep sobre docs devuelve cada match dos veces. Se corrige
excluyendo `.claude/worktrees/` de las búsquedas, no borrando el worktree (tiene commits propios sin
integrar).

## Verificación

Falsificable con la misma medición que produjo el 90.130: leer `cache_creation_input_tokens` del
primer request de la sesión siguiente en `~/.claude/projects/-Users-yo-Documents-kza/*.jsonl`.

Aritmética: 90.130 − 40.700 (MCP) − 6.400 (skills) − 3.000 (MEMORY.md) = **~40.000**.

- **Objetivo:** ~40.000 tokens; se acepta hasta 45.000, porque dos sumandos son estimaciones y no
  mediciones: claude-in-chrome (~4.000, no se pudo sondear por stdio como los otros) y el ahorro de
  skills (6.400 asume que el bloque de listado encoge proporcional a los frontmatter).
- Si queda por encima de 50.000, algo no se aplicó — revisar qué servers siguen cargando con `/mcp`.

Verificación secundaria, antes de cerrar: confirmar en konsi y homelab-infra que sus MCP siguen
respondiendo tras la reubicación.

## Reversibilidad

| Cambio | Cómo se revierte |
|---|---|
| MCP reubicados | backup de `~/.claude.json` previo al cambio |
| Skills archivadas | `mv ~/.claude/skills.archive/* ~/.claude/skills/` |
| CLAUDE.md, MEMORY.md | git (MEMORY.md no está en git — se copia a `MEMORY.md.bak` antes) |

## Riesgos

- **Un server reubicado al proyecto equivocado.** Ese proyecto pierde la herramienta hasta
  re-declararla. Acotado: la asignación sale de llamadas medidas, no de suposiciones, y el arreglo es
  volver a agregar la entrada.
- **Una skill archivada que se necesitaba.** El histórico cubre hasta 2026-08-03; una skill instalada
  para algo aún no empezado no tiene uso registrado y se archivaría. Mitigación: el archivo queda en
  disco y `find-skills` sigue disponible para reinstalar.
- **MEMORY.md pierde un gancho que importaba.** Mitigación: se poda solo lo marcado ✅ y ningún
  archivo de memoria se borra.
