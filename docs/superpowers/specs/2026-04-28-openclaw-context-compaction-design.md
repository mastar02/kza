# OpenClaw Plan #2 — Auto-compaction de Contexto + Identifier Policy Strict

> **Tipo:** Design spec. El plan implementable se genera después con `superpowers:writing-plans`.
> **Origen:** Item #2 del roadmap `docs/superpowers/plans/2026-04-26-openclaw-roadmap.md`. Patrón inspirado en `docs/concepts/compaction.md` de OpenClaw.
> **Fecha:** 2026-04-28.

## Goal

Reducir latencia y coste por prompt acumulado en sesiones largas (in-memory compaction) y dar memoria cross-sesión a usuarios recurrentes (snapshot persistido), sin nunca corromper identificadores opacos de Home Assistant en el proceso.

## Problemas que resuelve

1. **Prompt creciente en CPU.** El razonador 30B (`kza-llm-ik` :8200) corre en CPU a ~63 tok/s. Cada turno acumulado en `conversation_history` agranda el prompt y empeora la latencia del slow path. Hoy se trunca duro a 10 turnos sin contexto preservado.
2. **Pérdida de contexto cross-sesión.** Con `inactive_timeout=300s` (5min), un usuario que vuelve después de la cena empieza desde cero. Preferencias estables no se retienen.
3. **Riesgo de corromper IDs HA.** Si en el futuro algún componente reescribe turnos (resumen LLM ingenuo), un `light.escritorio_principal` mal renderizado deja la luz inoperante.

## No-goals (YAGNI)

- Vector search sobre summaries (plan #2.5 si se necesita).
- Incremental summary update con diff (cada compaction concatena al anterior).
- GC automático de archivos `data/contexts/` viejos.
- Multi-zona splits del contexto (un user → un summary).
- Re-compactación del summary cuando crece.

## Decisiones de diseño

| # | Decisión | Alternativa descartada | Razón |
|---|----------|------------------------|-------|
| D1 | Trigger in-memory: turn-count proactivo en background | Token-budget reactivo / idle opportunistic | Evita stall en turno activo (objetivo <300ms fast path). Idle no cubre usuarios verbosos. |
| D2 | Trigger persistencia: snapshot al expirar | Persistencia continua / vector ChromaDB | Mínimo viable. Vector queda para plan #2.5 si la persistencia simple no alcanza. |
| D3 | Identifier policy: extracción literal pre-compaction a campo `preserved_ids` | Post-hoc validation con retry / prompt instruction | Invariante por construcción. Los IDs nunca pasan por el modelo, no hay nada que romper. |
| D4 | Consumo de `preserved_ids`: hint a NLU regex + VectorSearch | Inyectar como system message al LLM | Mantener prompt LLM limpio de ruido estructurado; los IDs viven donde se resuelven. |
| D5 | Output del modelo en JSON (`{"summary": "..."}`) | Texto plano | Parser robusto, evita prefijos de razonamiento del modelo. Fallback a texto trimmed si JSON malformado. |
| D6 | Compaction NO usa LLMRouter (failover plan #1) | Sí usar LLMRouter | Compaction es background no-crítico. Si el 30B falla, prefiero silencio antes que ocupar el fast router 7B en una tarea no urgente. |
| D7 | Refactor mínimo de `cleanup_inactive` thread→asyncio task | Mantener thread + `run_coroutine_threadsafe` | El compactor es async; el callsite tiene que ser async para evitar acrobacias de loop. ~30 líneas, API pública intacta. |
| D8 | Default `enabled: false` en config | Default `enabled: true` | Deploy seguro. Activar tras validar en prod con un usuario controlado. |

## Arquitectura

```
                            ┌─────────────────┐
Turn N (proactivo) ─────────│                 │
Cleanup inactive (snapshot) ─│   Compactor    │── async ──> kza-llm-ik :8200 (30B)
Manual (tests/admin) ───────│                 │
                            └────────┬────────┘
                                     │ CompactionResult
                                     │
                            ┌────────▼────────┐         ┌──────────────────┐
                            │ ContextManager  │─ save ──│ ContextPersister │── data/contexts/<id>.json
                            │ (modificado)    │─ load ──│                  │
                            └────────┬────────┘         └──────────────────┘
                                     │ preserved_ids
                                     ▼
                          request_router → VectorSearch + NLU regex extractor
```

### Componentes

**Nuevos:**

- `src/orchestrator/compactor.py` — `Compactor`, `CompactionResult`. Recibe `HttpReasoner` por DI. Llama `await reasoner.complete(...)` con `response_format: json_object`.
- `src/orchestrator/context_persister.py` — `ContextPersister`. Sync file I/O con write atómico (rename `.tmp` → final).

**Modificados:**

- `src/orchestrator/context_manager.py` — campos nuevos en `UserContext`, hooks en `add_turn` y `cleanup_inactive`, hidratación en `get_or_create`. `cleanup_inactive` pasa de thread daemon a asyncio task.
- `src/main.py` — DI del Compactor + Persister al ContextManager.
- `src/pipeline/request_router.py` — pasa `ctx.preserved_ids` al VectorSearch y al regex extractor.
- `src/orchestrator/__init__.py` — exports.
- `config/settings.yaml` — bloque `context.compaction` y `context.persistence`.

## Flujos de datos

### Flujo A — Compactación in-memory (turn-count proactivo)

```
add_turn(user_id, role, content, intent, entities)  # turn N
└─ ctx.add_turn(...)                                 # historial pasa de N-1 a N
└─ if N == compaction_threshold (6) and not ctx.compaction_inflight:
       ctx.compaction_inflight = True
       asyncio.create_task(self._compact_background(user_id))

_compact_background(user_id):
    1. Bajo lock: copia los primeros (N - keep_recent=3) turnos a compactar
    2. Suelta el lock. Llama compactor.compact(turns, preserved_entities=union(t.entities))
    3. Bajo lock de nuevo:
        ctx.compacted_summary = (ctx.compacted_summary + " " if existe) + result.summary
        ctx.preserved_ids = unique(ctx.preserved_ids + result.preserved_ids)
        ctx.conversation_history = ctx.conversation_history[-keep_recent:]
        ctx.compaction_inflight = False
    4. Log: [Compactor] user=X turns=K summary_chars=Y preserved_ids=Z latency=Wms
```

**Threshold:** `compaction_threshold=6`, `keep_recent=3`. Disparo natural en turno 6, 12, 18 (si la session sigue larga). Configurable en `settings.yaml`.

### Flujo B — Snapshot a disco (cleanup_inactive)

```
cleanup_loop (asyncio task) detecta user X inactivo > inactive_timeout:
└─ if ctx tiene historia o summary:
       await _snapshot_and_remove(X)

_snapshot_and_remove(X):
    1. Si hay turnos no-compactados, await compactor.compact(restantes)
       (suma al summary existente; preserved_ids actualizado)
    2. persister.save(ctx) → data/contexts/<user_id>.json (write atómico)
    3. del self._contexts[X]
```

### Flujo C — Hidratación (próxima sesión)

```
get_or_create(user_id, ...):
└─ if user_id in self._contexts: return existente
└─ if persister.exists(user_id):
       data = persister.load(user_id)
       ctx.compacted_summary = data["compacted_summary"]
       ctx.preserved_ids = data["preserved_ids"]
       ctx.session_count = data["session_count"] + 1
       # conversation_history queda [] — los turnos viejos murieron
       log [ContextManager] hydrated user=X session_count=N
   else:
       ctx fresh, session_count=1
```

## Interfaces

### Compactor

```python
@dataclass
class CompactionResult:
    summary: str
    preserved_ids: list[str]
    compacted_turns_count: int
    model: str
    latency_ms: float

class Compactor:
    def __init__(
        self,
        reasoner: HttpReasoner,
        max_summary_tokens: int = 200,
        timeout_s: float = 30.0,
    ):
        ...

    async def compact(
        self,
        turns: list[ConversationTurn],
        preserved_entities: list[str],
    ) -> CompactionResult:
        ...
```

### Prompt al 30B

```
[system]
Sos un compactador de contexto conversacional. Recibís N turnos de diálogo
entre un usuario y un asistente de hogar. Tu tarea: producir un resumen en
2-4 oraciones en español que capture (a) preferencias estables del usuario,
(b) decisiones tomadas, (c) entidades del hogar referenciadas en lenguaje
natural (NO uses identificadores técnicos). Usá tercera persona.
NO menciones IDs tipo light.X, scene.Y, area.Z — esos se preservan aparte.

[user]
Turnos a compactar:
1. [user] Prendé la luz del escritorio
2. [assistant] Listo
...

Devolvé JSON: {"summary": "..."}
```

### UserContext (campos nuevos)

```python
@dataclass
class UserContext:
    # ... campos existentes
    compacted_summary: str | None = None
    preserved_ids: list[str] = field(default_factory=list)
    compaction_inflight: bool = False  # transient, no se serializa
    session_count: int = 1
```

### ContextPersister

```python
class ContextPersister:
    def __init__(self, base_path: Path = Path("data/contexts")):
        ...

    def save(self, ctx: UserContext) -> None:
        # write atómico: <id>.json.tmp + os.replace
        ...

    def load(self, user_id: str) -> dict | None:
        ...

    def exists(self, user_id: str) -> bool:
        ...
```

### Formato JSON persistido

```json
{
  "user_id": "user_juan",
  "user_name": "Juan",
  "last_seen": 1714349123.4,
  "session_count": 7,
  "compacted_summary": "El usuario prefiere iluminación tenue de noche...",
  "preserved_ids": ["light.escritorio_principal", "scene.modo_lectura"],
  "version": 1
}
```

Campo `version` para futuras migraciones de schema.

## Configuración (`config/settings.yaml`)

```yaml
context:
  compaction:
    enabled: false                  # OFF hasta validar en prod
    threshold_turns: 6
    keep_recent_turns: 3
    max_summary_tokens: 200
    timeout_s: 30.0
  persistence:
    enabled: false                  # OFF hasta validar en prod. Requiere compaction.enabled=true.
    base_path: "data/contexts"
    inactive_timeout_s: 300         # movido desde el constructor de ContextManager
```

Con ambos `enabled: false`, comportamiento idéntico al actual: truncate duro a `max_history=10`, cleanup borra sin persistir. **Backward-compatible.**

**Dependencia:** `persistence.enabled=true` requiere `compaction.enabled=true`. Persistir un historial sin compactar tendría poco sentido (se llenaría de turnos crudos repetidos cross-sesión). Validar al arrancar; loguear error y tratar persistence como off si la combinación es inválida.

**No hay knob de config para el endpoint del Compactor**: el `HttpReasoner` apuntando a kza-llm-ik se inyecta por código en `main.py` (consistente con D6 — Compaction no usa LLMRouter).

## Failure handling

| Falla | Manejo |
|-------|--------|
| `compactor.compact()` timeout (>30s) | Log error, `compaction_inflight=False`, historial intacto. Próximo trigger reintenta. |
| `compactor.compact()` HTTP error | Idem. Sin retry inmediato. |
| `compactor.compact()` JSON malformado | Fallback a texto literal trimmed como summary. Log warning. |
| `kza-llm-ik` cooldown del LLMRouter | NO usamos LLMRouter (D6). Falla directa, mismo handling que HTTP error. |
| `persister.save()` file I/O error | Log error, contexto se pierde. NO rompe cleanup. |
| `persister.load()` JSON corrupto | Log warning, tratar como inexistente. Contexto fresh. |

## Testing

**Unit tests nuevos** (≥ 25 casos):

- `test_compactor.py` — JSON válido / malformado / timeout / preserved_entities passthrough.
- `test_context_persister.py` — save+load round-trip, atomic rename, file no existe, JSON corrupto.
- `test_context_manager_compaction.py`:
  - Trigger en turno N exactamente una vez (inflight prevent dup).
  - Compaction success → historial reducido, summary set, IDs unionados.
  - Compaction failure → historial intacto, inflight=False, próximo trigger ok.
  - Múltiples compactions concatenan summaries y deduplican IDs.
  - `cleanup_inactive` activo → no persiste; expirado → sí persiste.
  - `get_or_create` con file existente hidrata.
  - `get_or_create` con file corrupto crea fresh + warning.
  - Backward-compat: `enabled=false` → comportamiento baseline idéntico.

**Integration test**:

- `test_context_compaction_e2e.py` — HttpReasoner mockeado a respuestas canónicas; flujo completo trigger → snapshot → hidratación.

## Outcome verificable post-deploy

1. Logs `[Compactor] user=X turns=3 summary_chars=187 latency=4200ms` aparecen en sesiones largas.
2. `data/contexts/<user>.json` existe post-cleanup, summary no vacío.
3. Próxima sesión del mismo user: log `[ContextManager] hydrated user=X session_count=2`.
4. Tests pasan (≥ 25 nuevos casos).
5. Con `enabled: false`, comportamiento idéntico a baseline (regression-safe).

## Próximo paso

Generar el plan implementable con `superpowers:writing-plans` desde este spec, luego ejecutar con `superpowers:subagent-driven-development` o `superpowers:executing-plans`.
