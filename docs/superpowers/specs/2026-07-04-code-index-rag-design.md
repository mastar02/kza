# Code-Index RAG — Diseño

**Fecha:** 2026-07-04
**Estado:** Aprobado (pendiente plan de implementación)
**Rama:** `feat/code-index`

## Problema

Analizar el proyecto completo (~38K líneas Python) con agentes profundos consume
una cantidad enorme de tokens: cada auditoría de bugs o exploración arranca de
cero con fan-out de Greps y lecturas exploratorias. Se necesita una estructura
de búsqueda que permita a los agentes localizar el código relevante de forma
directa y barata.

## Objetivo

Un índice semántico del codebase, consultable por los agentes de Claude Code,
que sirva para:

1. **Auditorías profundas** — llegar directo a los archivos/funciones candidatos
   sin releer todo el repo.
2. **Navegación diaria** — abaratar la fase de localización de cualquier tarea
   (buscar, entender, modificar).

**Expectativa explícita:** el ahorro está en la *localización*. Para auditar un
bug los agentes igual leen el código real de los candidatos; el índice elimina
la exploración, no la lectura final.

## Decisiones tomadas (con el usuario, 2026-07-04)

| Decisión | Elección |
|----------|----------|
| Caso de uso | Auditorías + navegación diaria (ambas) |
| Ubicación | Servicio en el server (reutiliza stack existente) |
| Contenido | Código chunked (AST) + resúmenes LLM ("cards") por archivo |
| Frescura | Hook post-deploy (`kza-sync` dispara reindex) |
| Arquitectura | Servicio dedicado `code-index`, aislado de producción |

## Arquitectura

```
laptop (agentes Claude Code)
   │  python tools/code_search.py "query"  →  HTTP a 192.168.1.2:9510
   ▼
code-index service :9510 (server, systemd --user kza)
 ├─ Chroma persistente propio  /home/kza/code-index/chroma/
 │    ├─ colección code_chunks  (funciones/clases/métodos)
 │    └─ colección code_cards   (resumen por archivo)
 ├─ BGE-M3 en CPU (embeddings, 0 VRAM)
 └─ indexer incremental (AST + cards vía gateway :8200 → MiniMax)
        ▲ trigger: curl POST /reindex al final de kza-sync
```

### Componentes

1. **Servicio HTTP `code-index`** — Python async (mismo estilo del proyecto),
   puerto **:9510** (sub-rango KZA 9500-9599; :9500 ya apuntado a obs/Quadlet
   Chroma — validar contra `docs/SERVER_CONVENTIONS.md` / Notion pág 8 antes de
   deploy). Endpoints:
   - `POST /search` — `{query, top_k, filters}` → resultados rankeados.
   - `POST /reindex` — `{mode: incremental|full}` → 202, indexa en background.
   - `GET /health` — estado + SHA indexado + timestamp del último reindex.
2. **Chroma persistente propio** en `/home/kza/code-index/chroma/`. Totalmente
   separado del Chroma in-process de kza-voice.
3. **Embeddings BGE-M3 en CPU** — carga lazy, `device="cpu"`. Cero VRAM; el
   Threadripper indexa 38K líneas en minutos y una query tarda ~100-200ms
   (irrelevante para búsqueda de código).
4. **Chunking por AST** — módulo `ast` de la stdlib (el proyecto es 100%
   Python). Un chunk por función/método/clase, con docstring, firma, path y
   rango de líneas como metadata.
5. **Cards por archivo con MiniMax** — vía gateway :8200 con virtual key propia
   del consumer `code-index`. Contenido de cada card: propósito, API pública,
   dependencias, invariantes/gotchas. Costo one-shot estimado ~400K tokens de
   entrada (~300 archivos); después solo incremental.

### Indexado incremental

- Manifest JSON con `git hash-object` por archivo (mismo patrón ya usado para
  medir drift laptop↔server).
- Reindex procesa solo archivos con hash cambiado: re-chunk + re-embed +
  regenerar card. Archivos borrados se purgan de ambas colecciones.
- El manifest registra el SHA de HEAD del árbol indexado.
- Escritura del manifest por archivo procesado → un reindex interrumpido es
  idempotente y retoma donde quedó.

### Consulta desde agentes

`tools/code_search.py` (CLI en el repo):

- Pega a `POST /search`, imprime: card resumida del módulo + `path:líneas` +
  snippet del chunk.
- **Warning de drift:** compara el hash local del archivo con el hash indexado;
  si difieren (rama de laptop no deployada), marca el resultado como STALE para
  que el agente lea el archivo real en lugar de confiar en el índice.
- Servicio caído → error claro y exit code ≠ 0; el agente cae de vuelta a
  Grep/Glob. El índice nunca es bloqueante.

### Alcance del indexado

- `src/**/*.py` (código productivo). No se indexan `tests/`, `docs/` ni
  `scripts/` en v1 (menos ruido; ampliable después si hace falta).

## Manejo de errores

| Falla | Comportamiento |
|-------|----------------|
| Servicio caído en query | CLI falla con mensaje claro; fallback manual a Grep |
| Gateway :8200 caído en reindex | Chunks se indexan igual; cards quedan pendientes y se reintentan en el próximo reindex |
| Reindex interrumpido | Manifest por-archivo → retoma incremental, sin corrupción |
| Drift laptop↔server | Hash-check en el CLI marca resultados STALE |

## Testing

- Unit tests: chunker AST (funciones anidadas, clases, decoradores, archivos
  vacíos), lógica incremental del manifest (agregado/modificado/borrado),
  formato de cards.
- Mocks de Chroma y gateway en `tests/mocks/` (patrón existente).
- Smoke test del endpoint `/search` y `/health`.

## Qué NO toca

- kza-voice, cuda:0, cuda:1 — cero impacto en producción de voz.
- El Chroma del pipeline (entidades HA).
- Ninguna reasignación de GPU (BGE-M3 del índice corre en CPU).

## Fuera de alcance (v1)

- Indexar tests/docs/scripts.
- MCP server como interfaz (el CLI alcanza; evaluable después).
- Timer nocturno adicional (el hook post-deploy cubre el flujo actual).
- Búsqueda híbrida BM25+denso (Chroma denso solo; recordar limitación conocida
  de BGE-M3 con antónimos — para código no aplica el caso "apagá/prendé").
