# KZA — Asistente de Voz Local para Domótica

> Detalle extenso (mapa de archivos/módulos, ejemplos de estilo, hardware, persistencia, comandos y guía multi-consola) en [`docs/referencia/guia-operativa.md`](docs/referencia/guia-operativa.md).

## Flujo de trabajo y despliegue acordado

- **Leer primero:** `AGENTS.md` y `docs/runbooks/2026-11-05-mac-development-handoff.md`.
- **Entorno Mac (autorizado 2026-11-05):** trabajar con la cuenta habitual sin privilegios y checkout local (`/Users/gabriel/Documents/Proyectos/kza`, `.venv` CPython 3.13); en Linux se usa `yo` en `/home/yo/projects/kza`.
- **Separación estricta de producción:** `kza` es la cuenta administrativa/runtime del servidor. **Nunca** editar código, instalar dependencias, compilar ni hacer `git pull` en `/home/kza/kza` o `/home/kza/app`, ni reiniciar/publicar servicios sin GO explícito de release.
- **Releases compiladas:** construir y probar artefactos de bytecode CPython 3.13 sin `.py` con `scripts/build_runtime.py` (`docs/architecture/DEPLOYMENT.md`). La release operativa `b0866e8` (2026-10-28) conserva estado privado separado y el checkout anterior para rollback.

## Resumen del Sistema

Sistema de control por voz local para Home Assistant. Latencia `<300ms` para domótica (fast path 100 % local); razonamiento complejo delegado al gateway LLM `:8200` (MiniMax cloud, decisión 2026-05-30 con `cloud.consent`). **2× RTX 3070 hoy:** `cuda:1` = STT + SpeakerID + TTS + `llama-server` 7B `:8101`; `cuda:0` = Emotion (deshabilitado) + fallback ambient (casi libre); BGE-M3 corre en CPU. Cualquier reasignación de GPUs se discute primero. Python 3.13, `async/await`, ~38K líneas, 8,900+ tests.

```
Mic → WakeWord(CPU) → STT(GPU1) → Router 7B(GPU1 :8101) → TTS(GPU1) → Speaker
                         ↕                  ↕
                   SpeakerID(GPU1)   Reasoner cloud (gateway :8200)
                   Emotion(GPU0)     ChromaDB / HomeAssistant
                   BGE-M3(CPU)
```

- **Paths:** Fast path (`<300ms`: HA → VectorSearch → TTS) · Music path (`~500ms`: Spotify → MoodMapper → ZoneController) · Slow path (gateway `:8200` → Memory → TTS).
- **Multi-usuario:** `MultiUserOrchestrator` → `PriorityRequestQueue` → `ContextManager` → `CancellationManager`.
- **Source of truth cross-project:** espejo local `docs/SERVER_CONVENTIONS.md` (puertos `9500-9599`, gateway `:8200`, Podman rootless). Notion se consulta solo bajo demanda; su MCP está retirado en KZA (no asumir `mcp__notion__*`). No leer memorias de `homelab-infra` ni `homelab-services`.

## Reglas de Ingeniería — LEER SIEMPRE

### SIEMPRE hacer
- `async/await` para toda operación I/O (HA, audio, red, disco).
- Inyección de dependencias por constructor; composición en lugar de herencia profunda.
- `@dataclass` para DTOs y `Enum` para estados/tipos (`ModelState`, `PathType`, `AlertPriority`).
- Docstrings Google-style y type hints en clases y funciones públicas.
- Orden de imports: stdlib → third-party → `from src.modulo import Clase` (nunca relativos).
- Logging con `logger = logging.getLogger(__name__)`; mensajes de voz/UI en español, código/logs en inglés.
- Tests con `.venv/bin/pytest` + fixtures en `tests/conftest.py`, mocks en `tests/mocks/`.
- Respetar asignación de GPUs (`cuda:1` audio+7B, `cuda:0` casi libre, BGE-M3 en CPU) y centralizar toda config en `config/settings.yaml`.
- Conservar la separación entre voz, persona, cuenta, roles, consentimiento y administración, y las releases gestionadas autocontenidas (`mem:decisions`).

### NUNCA hacer
- Bloquear el event loop con I/O síncrono ni usar `print()` en código de aplicación.
- Cambiar asignación de GPUs, agregar dependencias pesadas sin justificar (8 GB VRAM/GPU) o crear archivos de config nuevos fuera de `config/settings.yaml`.
- Modificar `src/main.py` sin revisar la cadena de DI completa.
- Hacer push directo a `main`, usar `git stash`, `git add -A` / `git add .`, ni commitear `.env`, `data/`, `models/`, secretos, audio o PII.
- Pedir confirmación cuando el siguiente paso de un plan autorizado es obvio; detenerse solo ante decisión ambigua, bloqueante real o acción destructiva.

## Workflow Multi-Consola (`herdr-app` + `hc` + `gpt-6-luna`)

- **Orquestador agnóstico (`claude`/`agy`/`codex`/`muse`):** mantiene roadmap, cierra decisiones previas con el usuario, despacha especialistas en worktrees aislados (`~/.herdr/worktrees/kza/<alias>`), juzga gates y mergea PRs. Nunca comparte un mismo worktree entre dos agentes ni implementa código de aplicación desde la raíz mientras coordina workers.
- **Pre-Task Spec Gate:** antes de cambios medianos/complejos, despachar `kza-analista` (read-only) para emitir Spec `<= 2 KB` con alcance, invariantes y decisiones abiertas.
- **Roles KZA (`herdr/templates/kza-*.md`):** `kza-analista` (spec), `kza-dev` (implementación), `kza-pregate` (pre-revisión rápida), `kza-qa` (gate independiente cross-model con 4 lentes: corrección/diseño, fallos silenciosos, rigor de tests, comentarios/afirmaciones), `kza-privacy` (autoridad/privacidad/releases autocontenidas), `kza-ui` (verificación browser/NEXO enrutada dinámicamente), `kza-ops` (packaging/CI/tooling, sin tocar prod).
- **Enrutamiento dinámico por cuotas reales:** `scripts/kza_router.py` consulta OpenAI Decisions API (`gpt-6-luna`, 0 output tokens) ponderado por `scripts/kza_quota.py` (`claude`, `codex`, `muse`, `agy`), con fallback determinista local y exclusión cross-model por **familia real** (`anthropic`|`openai`|`meta`|`google`). Nunca hardcodear un modelo único; actualizar capacidades/benchmarks en el catálogo cuando salgan modelos nuevos.
- **Asientos y notificación push sin polling:** abrir y cerrar vía `scripts/kza_seat.py` (`--label project=kza`, tab renombrado en `herdr-app` sin robar foco, `dirty=0` en detached). Los workers escriben su reporte en `~/.local/state/kza-router/reports/<alias>.md` y envían `<= 6` líneas con `hc send $HC_TERMINAL_ID`. El orquestador espera con `hc wait --children any` y verifica reporte, `HEAD` exacto y PR antes de aceptar `done`.
- **Aislamiento de proyectos y confianza:** jamás tocar terminales, templates (`tu-*`, `gsm-*`) ni worktrees de otros proyectos. Nunca bypassear diálogos de confianza de `agy`/`codex`/`muse`; aprobarlos en el tab visible de `herdr-app`.
- **Hooks y skills portables:** safety guard (`scripts/kza_safety_guard.py`) + `PreCompact` (`scripts/kza_precompact.py`) activos en `.claude/settings.json` y `.codex/hooks.json`; skills portables en `.agents/skills/kza-*` (enlazados en `.claude/skills/`).

## Comandos Rápidos

```bash
# Tests (Mac .venv CPython 3.13)
.venv/bin/pytest tests/unit/workflow/
.venv/bin/pytest tests/ -k "test_speaker"

# Orquestación multi-consola
python3 scripts/kza_quota.py --json
python3 scripts/kza_router.py route --role kza-qa --exclude-family google --summary "Revisar PR"
python3 scripts/kza_seat.py sync-templates
python3 scripts/kza_seat.py open --alias kza-qa-pr1 --role kza-qa --ref HEAD --exclude-family google --task "..."
python3 scripts/kza_seat.py status
python3 scripts/kza_seat.py close --alias kza-qa-pr1 --done-readonly
```
