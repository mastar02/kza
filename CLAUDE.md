# Nexa v2 — ingeniería y operación

> Detalle extenso (mapa de archivos/módulos, ejemplos de estilo, hardware, persistencia, comandos y guía multi-consola) en [`docs/referencia/guia-operativa.md`](docs/referencia/guia-operativa.md).

## Flujo de trabajo y despliegue acordado

- **Leer primero:** `AGENTS.md` y `docs/runbooks/2026-11-05-mac-development-handoff.md`.
- **Entorno Mac (autorizado 2026-11-05):** trabajar con la cuenta habitual sin privilegios y checkout local (`/Users/gabriel/Documents/Proyectos/kza`, `.venv` CPython 3.13); en Linux se usa `yo` en `/home/yo/projects/kza`.
- **Separación estricta de producción:** `kza` es la cuenta administrativa/runtime del monolito histórico. **Nunca** editar código, instalar dependencias, compilar ni hacer `git pull` en `/home/kza/kza` o `/home/kza/app`, ni reiniciar/publicar servicios sin GO explícito de release.
- **Releases nativas:** cuatro binarios Rust, worker Python 3.13 y SPA. Empaquetar y promover según el [contrato de despliegue](docs/architecture/DEPLOYMENT.md). La ejecución de Nexa corresponde al usuario dedicado `nexa`.

## Resumen del sistema

Cinco daemons nativos coordinan captura por habitación, inferencia offline, memoria SQLite, decisiones domésticas y NEXO. La configuración usa tres TOML tipados. La cuenta de la SPA y su rol autorizan las operaciones. El consentimiento limita las salidas cloud. Consultar [roadmap](docs/architecture/NEXA_V2_ROADMAP.md) e [inventario de hardware](docs/architecture/HARDWARE.md).

- **Contrato compartido:** conservar el espejo local de Notion en `docs/SERVER_CONVENTIONS.md`. Consultar Notion bajo demanda, sin asumir su MCP ni leer memorias de homelab-infra u homelab-services.

## Reglas de Ingeniería — LEER SIEMPRE

### SIEMPRE hacer
- `async/await` para toda operación I/O (HA, audio, red, disco).
- Inyección de dependencias por constructor; composición en lugar de herencia profunda.
- `@dataclass` para DTOs y `Enum` para estados/tipos (`ModelState`, `PathType`, `AlertPriority`).
- Docstrings Google-style y type hints en clases y funciones públicas.
- Orden de imports: stdlib → third-party → `import absoluto del paquete` (nunca relativos).
- Logging con `logger = logging.getLogger(__name__)`; mensajes de voz/UI en español, código/logs en inglés.
- Tests con `.venv/bin/pytest` (`tests-v2/`, `tools/nexa-migrate/tests`, `workers/nexa-ml`), `cargo nextest --workspace` y Vitest/Playwright en `frontend/`.
- Respetar asignación de GPUs (`cuda:1` LLM local `Qwen3.5-9B`, `cuda:0` STT comando+ambient Whisper compartido + Kokoro TTS + ECAPA, `BGE-M3` en CPU) y centralizar la configuración en los tres TOML de Nexa.
- Conservar la separación entre voz, persona, cuenta, roles, consentimiento y administración, y las releases gestionadas autocontenidas (`mem:decisions`).

### NUNCA hacer
- Bloquear el event loop con I/O síncrono ni usar `print()` en código de aplicación.
- Cambiar asignación de GPUs, agregar dependencias pesadas sin justificar (8 GB VRAM/GPU) o crear archivos de config nuevos fuera del contrato tipado.
- Hacer push directo a `main`, usar `git stash`, `git add -A` / `git add .`, ni commitear `.env`, `data/`, `models/`, secretos, audio, PII o `casa.glb`.
- Pedir confirmación cuando el siguiente paso de un plan autorizado es obvio; detenerse solo ante decisión ambigua, bloqueante real o acción destructiva.

## Workflow Multi-Consola (`herdr-app` + `hc` + `gpt-6-luna`)

- **Orquestador agnóstico (`claude`/`agy`/`codex`/`muse`):** mantiene roadmap, cierra decisiones previas con el usuario, despacha especialistas en worktrees aislados (`~/.herdr/worktrees/kza/<alias>`), juzga gates y mergea PRs. Nunca comparte un mismo worktree entre dos agentes ni implementa código de aplicación desde la raíz mientras coordina workers.
- **Pre-Task Spec Gate:** antes de cambios medianos/complejos, despachar `kza-analista` (read-only) para emitir Spec `<= 2 KB` con alcance, invariantes y decisiones abiertas.
- **Roles KZA (`herdr/templates/kza-*.md`):** `kza-analista` (spec), `kza-dev` (implementación), `kza-pregate` (pre-revisión rápida), `kza-qa` (gate independiente cross-model con 4 lentes: corrección/diseño, fallos silenciosos, rigor de tests, comentarios/afirmaciones), `kza-privacy` (autoridad/privacidad/releases autocontenidas), `kza-ui` (verificación browser/NEXO enrutada dinámicamente), `kza-ops` (packaging/CI/tooling, sin tocar prod).
- **Enrutamiento dinámico por cuotas reales:** `scripts/kza_router.py` consulta OpenAI Decisions API (`gpt-6-luna`, 0 output tokens) ponderado por `scripts/kza_quota.py` (`claude`, `codex`, `muse`, `agy`), con fallback determinista local y exclusión cross-model por **familia real** (`anthropic`|`openai`|`meta`|`google`). **`codex-2` tiene exclusión dura por debajo del 30% de cuota restante** (`>= 70%` uso en 5h o 7d) para no consumir la suscripción del compañero. Nunca hardcodear un modelo único; actualizar capacidades/benchmarks en el catálogo cuando salgan modelos nuevos.
- **Asientos y notificación push sin polling:** abrir y cerrar vía `scripts/kza_seat.py` (`--label project=kza`, tab renombrado en `herdr-app` sin robar foco, `dirty=0` en detached). Los workers escriben su reporte en `~/.local/state/kza-router/reports/<alias>.md` y envían `<= 6` líneas con `hc send $HC_TERMINAL_ID`. El orquestador espera con `hc wait --children any` y verifica reporte, `HEAD` exacto y PR antes de aceptar `done`.
- **Aislamiento de proyectos y confianza:** jamás tocar terminales, templates (`tu-*`, `gsm-*`) ni worktrees de otros proyectos. Nunca bypassear diálogos de confianza de `agy`/`codex`/`muse`; aprobarlos en el tab visible de `herdr-app`.
- **Hooks y skills portables:** safety guard (`scripts/kza_safety_guard.py`) + `PreCompact` (`scripts/kza_precompact.py`) activos en `.claude/settings.json` y `.codex/hooks.json`; skills portables en `.agents/skills/kza-*` (enlazados en `.claude/skills/`).

## Comandos Rápidos

```bash
# Tests (Mac .venv CPython 3.13)
.venv/bin/pytest tests-v2/workflow/
.venv/bin/pytest tests-v2/smoke tools/nexa-migrate/tests

# Orquestación multi-consola
python3 scripts/kza_quota.py --json
python3 scripts/kza_router.py route --role kza-qa --exclude-family google --summary "Revisar PR"
python3 scripts/kza_seat.py sync-templates
python3 scripts/kza_seat.py open --alias kza-qa-pr1 --role kza-qa --ref HEAD --exclude-family google --task "..."
python3 scripts/kza_seat.py status
python3 scripts/kza_seat.py close --alias kza-qa-pr1 --done-readonly
```
