---
name: kza-pregate
description: KZA pre-revisión rápida read-only antes del gate QA cross-model. Detecta higiene de diff, tests faltantes y violaciones obvias.
command: codex --no-daemon -m gpt-6.2-astra -c model_reasoning_effort=medium -a never -s danger-full-access --dangerously-bypass-hook-trust
---

# KZA `kza-pregate` — Pre-Gate Rápido (Read-Only)

Verificar `git rev-parse HEAD && git status --porcelain --untracked-files=no` (`dirty=0`). No editar archivos.

- **Alcance rápido:**
  1. Verificar que el diff cumple la Spec sin cambios fuera de alcance ni ruido.
  2. Detectar violaciones obvias de invariantes KZA (I/O bloqueante, `print()` en `src/`, secretos/PII, rutas `/home/kza/*`, dependencias injustificadas, mocks huecos).
  3. Confirmar que los tests nuevos y afectados corren y pasan en `.venv/bin/pytest`.
- **Veredicto:** emitir `LISTO_PARA_QA` o `CORREGIR` con lista breve y accionable (archivo:línea).
- **Entrega:** escribir reporte UNA vez en `report_path` y enviar sobre `<= 6` líneas vía `hc send <parent>`.
