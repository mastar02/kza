---
name: kza-worktree-hygiene
description: Reglas de aislamiento de worktrees, diálogos de confianza en tabs visibles y cierre seguro de asientos KZA.
---

# KZA Worktree & Trust Hygiene

## Aislamiento
- Cada tarea especialista corre en su propio worktree bajo `~/.herdr/worktrees/kza/<alias>`.
- Nunca ejecutar dos agentes concurrentes sobre el mismo checkout o el mismo worktree.
- Worktrees detached (`kza-analista`, `kza-pregate`, `kza-qa`, `kza-privacy`, `kza-ui`) deben iniciar con `dirty=0` (nunca sobrescribir archivos trackeados).
- Si `hc run` falla, `kza_seat.py` hace rollback automático del worktree y la rama recién creada.

## Confianza Inicial (Nunca Bypassear)
- `agy` (primera vez en un worktree nuevo), `muse` (workspace nuevo) y `codex` (primera vez con `.codex/hooks.json`) se pausan en estado `blocked` pidiendo confirmación humana.
- **Nunca** bypassear diálogos de confianza programáticamente. El humano aprueba directamente en el tab visible del space `kza` en `herdr-app`.

## Cierre Seguro (`kza_seat.py close`)
- Nunca cerrar un asiento en `working` o `blocked` sin `--abandon`.
- Verificar que no haya suciedad ajena fuera de `ensure_manifest` (incluyendo `.serena/memories/*`).
- Roles de escritura: exigir `--merged` (valida `git merge-base --is-ancestor`) o `--merged-squash`.
- Roles de lectura: exigir `--done-readonly`.
- Conservar intacto el worktree hermano `bench-ambient` y nunca tocar worktrees/templates de otros proyectos (`tu-*`, `gsm-*`).
- Una vez cerrado el asiento con `kza_seat.py close` y validados el merge (si corresponde) y la evidencia, borrar `~/.local/state/kza-router/reports/<alias>.md` y limpiar las cachés `target/` de KZA que ya no se necesiten.
