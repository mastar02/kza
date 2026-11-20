---
name: kza-orchestrator
description: Playbook portable del Orquestador KZA multi-consola (Claude, Codex, Muse, Agy) usando herdr-app, hc, kza_quota.py, kza_router.py y kza_seat.py.
---

# KZA Multi-Console Orchestrator

Cualquier consola principal (`claude`, `agy`, `codex`, `muse`) en el space `kza` de `herdr-app` actúa como **Orquestador**.

## Reglas de Oro
1. **Separación de rol:** el orquestador mantiene el roadmap, cierra decisiones con el usuario, despacha especialistas en worktrees aislados, juzga gates y mergea PRs. No implementa código de aplicación en la raíz mientras hay workers activos.
2. **Pre-Task Spec Gate:** antes de tareas medianas/complejas, abrir `kza-analista` (read-only) para emitir Spec `<= 2 KB` y cerrar decisiones abiertas con el usuario.
3. **Enrutamiento dinámico por cuotas reales:** usar siempre `scripts/kza_router.py` (`gpt-6-luna`) + `scripts/kza_quota.py` — nunca hardcodear un modelo único (incluido `kza-ui`). En gates `kza-qa` / `kza-privacy`, pasar `--exclude-family <familia_del_autor>` para cross-model estricto por **familia real** (`anthropic` | `openai` | `meta` | `google`).
4. **Push sin polling:** esperar reactivamente con `hc wait --children any`. Nunca aceptar `done` sin verificar:
   - archivo de reporte en `~/.local/state/kza-router/reports/<alias>.md` (o salida inline si Muse `--disable-write`),
   - `HEAD` exacto del worktree/PR,
   - estado de PR y tests reales.
5. **Aislamiento total:** operar solo sobre `--label project=kza`, worktrees `~/.herdr/worktrees/kza/<alias>` y templates `kza-*.md`. Jamás tocar `tu-*`, `gsm-*` ni producción `/home/kza/*`.

## Flujo Operativo
```bash
# 1. Revisar cuotas reales
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

# 4. Verificar reporte y cerrar con guardas
python3 scripts/kza_seat.py status
python3 scripts/kza_seat.py close --alias kza-qa-pr42 --done-readonly

# 5. Registrar outcome para KPIs (>=85% Ronda 1)
python3 scripts/kza_router.py outcome <decision_id> --rounds 1 --approved true
python3 scripts/kza_router.py report --days 30
```
