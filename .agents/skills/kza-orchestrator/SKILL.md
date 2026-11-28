---
name: kza-orchestrator
description: Playbook portable del Orquestador KZA multi-consola (Claude, Codex, Muse, Agy) usando herdr-app, hc, kza_quota.py, kza_router.py y kza_seat.py.
---

# KZA Multi-Console Orchestrator

Cualquier consola principal (`claude`, `agy`, `codex`, `muse`) en el space `kza` de `herdr-app` actúa como **Orquestador**.

## Reglas de Oro
1. **Separación de rol:** el orquestador mantiene el roadmap, cierra decisiones con el usuario, despacha especialistas en worktrees aislados, juzga gates y mergea PRs. No implementa código de aplicación en la raíz mientras hay workers activos.
2. **Pre-Task Spec Gate:** antes de tareas medianas/complejas, abrir `kza-analista` (read-only) para emitir Spec `<= 2 KB` y cerrar decisiones abiertas con el usuario.
3. **Enrutamiento dinámico por cuotas reales:** usar siempre `scripts/kza_router.py` (`gpt-6-luna`) + `scripts/kza_quota.py` sobre los 4 runtimes / 6 pools activos (`claude-1`, `codex` + `codex-2` sin `/fast` y con **reserva obligatoria >= 30% restante en `codex-2`**, exclusión dura `>= 70%` uso en 5h/7d para preservar la suscripción del compañero, `muse`, `agy:gemini` + `agy:3p` vía tap `quota-agy.json` / `agy -p /usage`) — nunca hardcodear un modelo único (incluido `kza-ui`). Para tareas largas, pasar `--routing-summary` corto a `kza_seat.py open` (o dejar que `distill_routing_summary` limpie SHAs/rutas/boilerplate antes de llamar a `gpt-6-luna`). En gates `kza-qa` / `kza-privacy`, `--exclude-family <familia_del_autor>` es obligatorio para cross-model estricto por **familia real** (`anthropic` | `openai` | `meta` | `google`).
4. **Push sin polling:** esperar reactivamente con `hc wait --children any`. Nunca aceptar `done` sin verificar:
   - archivo de reporte en `~/.local/state/kza-router/reports/<alias>.md` (validado automáticamente no vacío en `kza_seat.py close`),
   - `HEAD` exacto del worktree/PR,
   - estado de PR y tests reales.
5. **Aislamiento total:** operar solo sobre `--label project=kza`, worktrees `~/.herdr/worktrees/kza/<alias>` y templates `kza-*.md`. Jamás tocar `tu-*`, `gsm-*` ni producción `/home/kza/*`.

## Flujo Operativo
```bash
# 1. Revisar cuotas reales (claude-1, codex/codex-2, muse, agy)
python3 scripts/kza_quota.py

# 2. Abrir asiento especialista (ruta dinámica + worktree + tab rename sin robar foco)
python3 scripts/kza_seat.py open \
  --alias kza-qa-pr42 \
  --role kza-qa \
  --ref HEAD \
  --exclude-family google \
  --task "Auditar PR #42 con 4 lentes..."

# 3. Esperar reactivamente (sin polling)
hc wait --children any

# 4. Verificar reporte (obligatorio al cerrar) y registrar outcome integrado para KPIs (>=85% Ronda 1)
python3 scripts/kza_seat.py status
python3 scripts/kza_seat.py close --alias kza-qa-pr42 --done-readonly --approved true --rounds 1
python3 scripts/kza_router.py report --days 30
```

Una vez cerrado el asiento con `kza_seat.py close` y validados el merge (si corresponde) y la evidencia, borrar `~/.local/state/kza-router/reports/<alias>.md` y limpiar las cachés `target/` de KZA que ya no se necesiten.

## Validación de Nexa v2

Ejecutar los checks desde el checkout del SHA en revisión. En un worktree Mac,
usar el venv Python 3.13 compartido y fijar `CARGO_TARGET_DIR` al directorio
de compilación acordado. Registrar SHA, comandos, resultados y skips.

- Para el orquestador y los hooks, ejecutar `pytest tests-v2/workflow -q`.
- Para Rust, ejecutar `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo nextest run --workspace` y `cargo nextest run -p nexa-web --features test-mode`.
- En `workers/nexa-ml`, ejecutar `python -m pytest -m "not gpu" -q` con las dependencias de desarrollo del worker.
- Para la SPA, ejecutar `npm --prefix frontend run check`, `npm --prefix frontend run test` y `npm --prefix frontend run build`. El gate L4 usa Chromium en `tests-v2/e2e` contra `nexa-web` compilado con `test-mode`.
- Para L5 y migración offline, compilar `nexa-memoryd`, fijar `NEXA_MEMORYD_BIN` al binario y ejecutar `pytest tests-v2/smoke tools/nexa-migrate/tests -q`.

La [CI v2](../../../.github/workflows/v2-ci.yml) define el entorno Linux y los pasos completos de cada gate.
