---
name: kza-orchestrator
description: Playbook portable del Orquestador KZA multi-consola (Claude, Codex, Muse, Agy) usando herdr-app, hc, kza_quota.py, kza_router.py y kza_seat.py.
---

# KZA Multi-Console Orchestrator

Cualquier consola principal (`claude`, `agy`, `codex`, `muse`) en el space `kza` de `herdr-app` actúa como **Orquestador**.

## Reglas de Oro
1. **Separación de rol:** el orquestador mantiene el roadmap, cierra decisiones con el usuario, despacha especialistas en worktrees aislados, juzga gates y mergea PRs. No implementa código de aplicación en la raíz mientras hay workers activos.
2. **Pre-Task Spec Gate:** antes de tareas medianas/complejas, abrir `kza-analista` (read-only) para emitir Spec `<= 2 KB` y cerrar decisiones abiertas con el usuario.
3. **Enrutamiento dinámico por cuotas reales:** usar siempre `scripts/kza_router.py` (`gpt-6-luna`) + `scripts/kza_quota.py` sobre los 4 runtimes activos (`claude` en cuenta `claude-1`, `codex` balanceado automáticamente entre `codex` y `codex-2` sin `/fast`, `muse`, `agy`) — nunca hardcodear un modelo único (incluido `kza-ui`). En gates `kza-qa` / `kza-privacy`, `--exclude-family <familia_del_autor>` es obligatorio para cross-model estricto por **familia real** (`anthropic` | `openai` | `meta` | `google`).
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

## Tiers de validación

Ejecutar desde el checkout del SHA que se está verificando con el venv KZA
de Python 3.13. En un worktree Mac sin venv propio, usar la ruta absoluta al
Python del venv compartido y mantener el directorio de trabajo del asiento.

- **Tier 0**, feedback de asiento con objetivo de 15–45 s. Seleccionar los
  tests afectados. Omitir `--no-node` si el cambio afecta al frontend.
- **Tier 1**, validación amplia en Mac o en el checkout de desarrollo de
  `kza-yo`. Ejecuta Node y pytest con hasta 8 workers por defecto. Excluye
  los cinco módulos de systemd real y los ocho casos `compiled-solo`.

```bash
.venv/bin/python scripts/run_test_suite.py --tier 0 --no-node --targets tests/unit/tools/test_run_test_suite.py
.venv/bin/python scripts/run_test_suite.py --tier 1
```

**Tier 2**, gate completo coordinado por el orquestador, como `yo` en
`/home/yo/projects/kza` de `kza-yo`. Reservar la ventana de pruebas y respetar
`SOLO_WINDOW` y `feature-pytest.lock` para evitar otras corridas durante
el gate. Mantener los seis workers y los límites de hilos y CPU.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  RAYON_NUM_THREADS=2 TOKIO_WORKER_THREADS=2 \
  taskset -c 24-47 nice -n 10 .venv/bin/python scripts/run_test_suite.py --tier 2 --workers 6
```

El Tier 2 ejecuta Node, el carril paralelo sin systemd y el carril serial
con systemd de usuario real. Al terminar los tres, ejecuta `compiled-solo`.
Aceptar el gate con `exit_code == 0`, `nrestarts_ok == true` y dos lecturas
válidas de journald que cumplan `before_NRestarts == after_NRestarts` en
`summary.json`. Incluir en el reporte del asiento el SHA, el tier, los
targets y la ruta de ese resumen. GitHub Actions verifica unit, safety e
integration con xdist y reserva los cinco módulos de systemd real para Tier 2.
