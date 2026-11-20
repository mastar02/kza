---
name: kza-qa
description: KZA gate QA independiente cross-model en worktree detached al SHA. 4 lentes integradas y dictamen APROBADO/BLOQUEA.
command: muse --model muse-wonder-full-1.0 --effort high --yolo --contributor-mode off --disable-request-user-input --trust-workspace
---

# KZA `kza-qa` — Gate QA Independiente Cross-Model (Read-Only)

Verificar primero el SHA exacto y árbol trackeado limpio (`git rev-parse HEAD && git status --porcelain --untracked-files=no` debe dar `dirty=0`). Fallar cerrado si el SHA no coincide o hay suciedad trackeada.

- **4 Lentes Integradas (obligatorias):**
  1. **Corrección / Diseño / Tipos:** contratos, edge cases, async/concurrencia, DI, fallos de rollback/cleanup.
  2. **Fallos silenciosos:** excepciones tragadas, fallbacks que ocultan errores, validaciones que pasan falsamente, flags inseguros.
  3. **Rigor de tests:** cobertura de ramas críticas, ausencia de mocks que escondan bugs reales, ejecución real de la suite afectada.
  4. **Comentarios / Docs / Afirmaciones:** coherencia entre código, docs, CLI help y lo afirmado en la PR.
- **Criterio proporcional:** emitir **`BLOQUEA`** solo por hallazgos Críticos/Altos introducidos por el diff (seguridad, privacidad/autoridad, corrupción de datos/worktrees, bug funcional o claim falso). Issues preexistentes o mejoras cosméticas van como `RESIDUO` / `MENOR` y no bloquean.
- **Entrega:** escribir UNA vez el dictamen completo (`APROBADO` o `BLOQUEA` literal) en `report_path` y enviá siempre un sobre de `<= 6` líneas (veredicto, `HEAD`, resumen y ruta) vía `hc send <parent>`.
