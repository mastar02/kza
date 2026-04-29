# OpenClaw Plan #3 — Plugin Hooks System Tipados

> **Tipo:** Design spec. El plan implementable se genera después con `superpowers:writing-plans`.
> **Origen:** Item #3 del roadmap `docs/superpowers/plans/2026-04-26-openclaw-roadmap.md`. Patrón inspirado en `docs/plugins/hooks.md` de OpenClaw (TS/Node, 30+ hooks tipados con priority + block/requireApproval/params-rewrite).
> **Fecha:** 2026-04-29.

## Goal

Un sistema de hooks tipados que permita a handlers Python escritos por el dueño del sistema (a) bloquear acciones de Home Assistant según políticas de seguridad y permisos, (b) reescribir parámetros de acciones HA y texto TTS antes de ejecutar, y (c) auditar todos los eventos relevantes del pipeline (wake / STT / intent / HA action / LLM / TTS) sin tocar el core. Backward-compatible: feature OFF por default, sin handlers el sistema corre como hoy.

## Use cases que motivan el plan

1. **Safety policies HA** — "Nunca desarmar la alarma de noche", "Cap brightness en cuarto del bebé". Hook `before_ha_action` con `block` o `rewrite`.
2. **Multi-user permission gating (block puro, sin override)** — "Los chicos no pueden controlar el clima ni la alarma". Hook `before_ha_action` con `block`. Override interactivo via `require_approval` queda fuera de scope (plan #3.5 si emerge).
3. **Audit / observability** — Cada evento del pipeline se persiste a SQLite (o syslog/webhook futuro). Use case: dashboard analytics, debugging post-hoc, training data collection.
4. **TTS rewriting** — Sanitización de texto antes del motor TTS (números, símbolos, abreviaturas). Hook `before_tts_speak` con `rewrite`.

## No-goals (YAGNI)

- `require_approval` — conversational confirmation flow. Diferido a plan #3.5.
- Plugin loading dinámico (file watcher, hot reload).
- Async handlers en `before_*` (sync only para proteger latencia <300ms del fast path).
- Hooks externos vía MCP / HTTP (KZA es local).
- Versioning de hook API (no es plataforma de plugins terceros).
- Per-user enable/disable (config global).
- Hooks `before_llm_call`, `before_compaction`, etc. — solo 3 hook points en este plan. La abstracción permite agregar más fácilmente cuando emerjan use cases.

## Decisiones de diseño

| # | Decisión | Alternativa descartada | Razón |
|---|---|---|---|
| D1 | Handlers son Python imperativo registrado vía decorador | DSL declarativo en YAML / híbrido | KZA es single-user/single-dev; iteración Python+pytest > YAML+restart cuando hay tests. Evita mantener un DSL que crece. |
| D2 | Outputs: `BlockResult` + `RewriteResult` solamente | También `RequireApproval` | `RequireApproval` es feature por sí sola (~150 líneas conversational state). Difiere para plan #3.5. |
| D3 | Handlers `before_*` son sync (`def`), `after_*` pueden ser sync o async | Todo async con timeout | Async en path crítico = riesgo de blow up de latencia <300ms. Sync force constraint. |
| D4 | `before_*` con priority numeric (lower runs first), `after_*` sin priority | All same priority / N priority levels | Permission gating debe correr antes que safety; granularidad numérica es cero overhead. |
| D5 | Block short-circuita la chain; rewrite chain encadena | Run all + collect | Block es "no quiero que pase"; correr handlers después no agrega info. Rewrite chain permite composición ortogonal. |
| D6 | Errors en handlers se loguean + cuentan, NO abortean | Re-raise / abort | "Política X bugueada" no debe romper la voice pipeline. Si querés abortar, retornás Block. |
| D7 | Eventos enumerados (lista cerrada) en `types.py` | Strings arbitrarios | Type-safe, autocompletable, documentación implícita. |
| D8 | Frozen dataclasses con `replace()` helper para rewrites | Mutables | Handler no puede corromper el estado por error. |
| D9 | `_global_registry` singleton + module-level decorator side effects | Pass registry explícito a cada handler | Decorador con efecto side al import es el pattern Python idiomático (Flask, Django). Single-process simplifica. |
| D10 | Default `enabled: false` | Default `true` | Backward-compat. Activar tras land. |

## Arquitectura

```
                                 ┌──────────────────┐
   src/policies/*.py    ───imp───>│  HookRegistry   │
   (decorators register)          │   (singleton)    │
                                 └────┬─────────────┘
                                      │ injected by main.py
                                      ▼
   ┌─────────────────────┐     ┌─────────────────────┐
   │ RequestDispatcher   │ ←── │  ResponseHandler   │
   │  before_ha_action   │     │  before_tts_speak  │
   └─────────────────────┘     └─────────────────────┘
              │                            │
              ▼ (block or call)            ▼
   ┌─────────────────────┐     ┌─────────────────────┐
   │  HomeAssistant      │     │  Kokoro TTS         │
   └─────────────────────┘     └─────────────────────┘

   Pipeline checkpoints emit after_event(name, payload):
   wake / stt / intent / ha_action_dispatched / ha_action_blocked / llm_call / tts
```

### Componentes

**Nuevos:**

- `src/hooks/types.py` — frozen dataclasses: `HaActionCall`, `TtsCall`, `BlockResult`, `RewriteResult`, payloads (`WakePayload`, `SttPayload`, `IntentPayload`, `HaActionDispatchedPayload`, `HaActionBlockedPayload`, `LlmCallPayload`, `TtsPayload`).
- `src/hooks/registry.py` — `HookRegistry` con `_before: dict[str, list[(prio, fn)]]`, `_after: dict[str, list[fn]]`, `_after_tasks: set[Task]`, counters.
- `src/hooks/decorators.py` — `before_ha_action(priority=100)`, `before_tts_speak(priority=100)`, `after_event(*event_names)`.
- `src/hooks/runner.py` — `execute_before_chain` (sync, block short-circuit + rewrite chain) + `execute_after_event` (fire-and-forget, sync handler inline + async handler como Task).
- `src/policies/{safety_alarm,permissions,audit_sqlite,tts_rewrite_es}.py` — las 4 policies que validan los use cases.

**Modificados:**

- `src/orchestrator/dispatcher.py` — acepta `hooks` por DI; antes de cada HA call invoca `execute_before_chain("before_ha_action", call)`.
- `src/pipeline/response_handler.py` — acepta `hooks` por DI; antes de cada `speak()` invoca `execute_before_chain("before_tts_speak", tts_call)`.
- `src/main.py` — si `hooks.enabled`: construye registry, importa `src.policies`, inyecta a dispatcher + response_handler, agrega `execute_after_event` calls en checkpoints.
- `config/settings.yaml` — bloque `hooks`.

## Flujos

### Flujo A — `before_ha_action` (path crítico)

```
RequestDispatcher decide ejecutar HA action:
  call = HaActionCall(entity_id, domain, service, service_data, user_id, user_name, zone_id, ts)
  result = registry.execute_before_chain("before_ha_action", call)
  if isinstance(result, BlockResult):
      response_handler.speak(result.reason)
      registry.execute_after_event("ha_action_blocked", HaActionBlockedPayload(...))
      return
  if isinstance(result, HaActionCall):  # rewrite happened
      call = result
  await ha_client.call_service(call)
  registry.execute_after_event("ha_action_dispatched", HaActionDispatchedPayload(...))
```

`execute_before_chain` itera handlers ordenados por priority ascendente. Cada handler es sync. Block short-circuita; rewrite encadena la `call` actualizada al siguiente handler. None pasa al siguiente sin cambios. Errores se loguean con `logger.warning`, contador `_handler_failures += 1`, chain continúa.

Convención de timing: handlers `before_*` deben ejecutar en <5ms. Validación opcional via `time.perf_counter` con `logger.warning` si superan `before_handler_warn_ms` (default 5.0). Sin abortar.

### Flujo B — `before_tts_speak` (path crítico)

Idéntico al A pero con `TtsCall(text, voice, lang, user_id, zone_id)`. RewriteResult modifica `text` antes de pasar al motor TTS. Block cancela el speak (raro pero válido) y loguea.

### Flujo C — `after_event` (fire-and-forget)

```
registry.execute_after_event("stt", SttPayload(...))
```

Internamente:
- Sync handlers: ejecutados directamente (rápido, no bloquea).
- Async handlers: `asyncio.create_task(handler(payload))`, agregada a `_after_tasks` con `add_done_callback(_after_tasks.discard)` para evitar GC.
- Errors: `task.add_done_callback` chequea exception y loguea + cuenta sin propagar.

## Interfaces

### Decorators

```python
# src/hooks/decorators.py

def before_ha_action(priority: int = 100):
    """Register a SYNC handler.

    Signature: def handler(call: HaActionCall) -> BlockResult | RewriteResult | None
    Lower priority runs first.
    """

def before_tts_speak(priority: int = 100):
    """Same as before_ha_action but for TtsCall."""

def after_event(*event_names: str):
    """Register sync OR async handler for one or more after-events.

    Signature: def/async def handler(payload) -> None
    Valid event_names enumerated in EVENT_NAMES (types.py).
    """
```

### Types

```python
@dataclass(frozen=True, slots=True)
class HaActionCall:
    entity_id: str
    domain: str
    service: str
    service_data: dict
    user_id: str | None
    user_name: str | None
    zone_id: str | None
    timestamp: float

    def with_data(self, **changes) -> "HaActionCall": ...


@dataclass(frozen=True, slots=True)
class TtsCall:
    text: str
    voice: str | None
    lang: str
    user_id: str | None
    zone_id: str | None


@dataclass(frozen=True, slots=True)
class BlockResult:
    reason: str
    rule_name: str


@dataclass(frozen=True, slots=True)
class RewriteResult:
    modified: object   # HaActionCall | TtsCall
    rule_name: str
```

Eventos: `EVENT_NAMES: tuple[str, ...] = ("wake", "stt", "intent", "ha_action_dispatched", "ha_action_blocked", "llm_call", "tts")`. Cada uno tiene su payload dataclass tipada.

### Registry

```python
class HookRegistry:
    def register_before(self, hook_name: str, fn, priority: int) -> None: ...
    def register_after(self, event_name: str, fn) -> None: ...

    def execute_before_chain(
        self, hook_name: str, call
    ) -> BlockResult | HaActionCall | TtsCall:
        """Returns BlockResult to short-circuit, or the (possibly rewritten) call.
        Sync — no await. Errors in handlers are logged + counted, never propagated."""

    def execute_after_event(self, event_name: str, payload) -> None:
        """Fire-and-forget. Sync handlers inline, async handlers as Tasks.
        Maintains strong refs in _after_tasks to prevent GC."""

    def get_stats(self) -> dict:
        """Returns _handler_failures, _handler_last_error, after_tasks_in_flight,
        before_handler_count_by_hook, after_handler_count_by_event."""

_global_registry: HookRegistry  # module-level singleton
```

### Las 4 policies de validación

`src/policies/safety_alarm.py`:
```python
@before_ha_action(priority=10)
def proteger_alarma_de_noche(call):
    if call.entity_id == "alarm_control_panel.casa" and call.service == "alarm_disarm":
        h = datetime.now().hour
        if h >= 22 or h < 7:
            return BlockResult(
                reason="No puedo desarmar la alarma a esta hora",
                rule_name="proteger_alarma_de_noche",
            )
    return None
```

`src/policies/permissions.py`:
```python
CHILD_USER_IDS = {"niño1", "niño2"}
ADULT_DOMAINS = {"climate", "lock", "alarm_control_panel"}

@before_ha_action(priority=5)
def chicos_sin_dominios_adultos(call):
    if call.user_id in CHILD_USER_IDS and call.domain in ADULT_DOMAINS:
        return BlockResult(
            reason="No tenés permiso para eso",
            rule_name="chicos_sin_dominios_adultos",
        )
    return None
```

`src/policies/audit_sqlite.py`:
```python
@after_event("ha_action_dispatched", "ha_action_blocked", "stt", "intent", "wake", "llm_call", "tts")
async def log_to_sqlite(payload):
    # Insert row in ./data/audit.db using asyncio.to_thread
    ...
```

`src/policies/tts_rewrite_es.py`:
```python
@before_tts_speak(priority=10)
def numeros_a_palabras(call: TtsCall):
    new_text = re.sub(r"\$(\d+)", r"\1 pesos", call.text)
    if new_text != call.text:
        return RewriteResult(
            modified=replace(call, text=new_text),
            rule_name="numeros_a_palabras",
        )
    return None
```

## Configuración

```yaml
hooks:
  enabled: false                   # OFF por default
  policies_dir: "src.policies"     # módulo Python a importar; los decoradores corren al import
  before_handler_warn_ms: 5.0      # log warning si un before_ handler supera este threshold
  audit_sqlite_path: "./data/audit.db"
```

Con `enabled: false`, `main.py` no importa `src.policies`, los decoradores no corren, el registry queda vacío, y dispatcher / response_handler reciben `hooks=None` (path no-op). Backward-compat garantizado.

## Failure handling

| Falla | Manejo |
|---|---|
| Handler `before_*` raisea exception | log warning, `_handler_failures += 1`, chain continúa |
| Handler `before_*` >5ms | log warning, NO aborta |
| Handler `after_event` async raisea | done_callback loguea + cuenta, descartado |
| Handler `after_event` sync raisea | log warning + cuenta |
| BlockResult con reason vacío | speak fallback "No puedo hacer eso" |
| Rewrite returns wrong type | log error, ignora rewrite, chain continúa |
| Import de `src.policies` falla | log error, registry queda vacío, sistema arranca sin hooks |

## Testing

**Unit (~25 nuevos):**

- `test_types.py` — frozen invariants, `with_data` helper, enum closeness.
- `test_registry.py` — register / list / clear / get_stats.
- `test_runner.py`:
  - Block short-circuit (handler 2 NO corre tras BlockResult del 1)
  - Rewrite chain (handler 2 ve la call modificada por handler 1)
  - Priority ordering (priority=5 corre antes que priority=10)
  - Handler exception → log + counter + chain sigue
  - After-event sync: ejecuta inline
  - After-event async: lanza Task, strong ref en `_after_tasks`, errors swallowed
  - Backward compat: `hooks=None` → no-op idéntico al baseline
- `test_safety_alarm.py` — block 22:00-06:59, no block 07:00-21:59
- `test_permissions.py` — block child+ADULT_DOMAIN, no block adult
- `test_tts_rewrite_es.py` — `$100` → `100 pesos`, no rewrite si no matchea

**Integration:**

- `test_hooks_e2e.py` — pipeline completo con todas las 4 policies activas, verificar:
  - "desarmá la alarma" a las 23:00 → block + TTS dice reason
  - "subí el clima" desde niño1 → block + log a audit.db
  - Comando con "$1000" → TTS habla "1000 pesos"
  - Audit SQLite tiene N rows tras N comandos

## Outcome verificable post-deploy

1. Tests pasan (≥ 25 nuevos)
2. Con `hooks.enabled: false` → behavior idéntico al baseline (regression-safe)
3. Con `hooks.enabled: true`:
   - "Nexa, desarmá la alarma" a las 23hs → TTS dice "No puedo desarmar la alarma a esta hora", `audit.db` registra `ha_action_blocked`
   - `audit.db` crece con eventos reales tras 1h de uso
   - Comando con `$1000` en TTS pronuncia "1000 pesos" en lugar de "dólar mil"
4. `HookRegistry.get_stats()` accesible vía `MultiUserOrchestrator.get_stats()` (mismo patrón que counters de plan #2). Counters expuestos: `_handler_failures`, `_handler_last_error`, `after_tasks_in_flight`, conteos por hook.

## Próximo paso

Generar el plan implementable con `superpowers:writing-plans` desde este spec, luego ejecutar con `superpowers:subagent-driven-development`.
